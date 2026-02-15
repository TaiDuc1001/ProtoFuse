import copy
import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils import (
    logger,
    setup_logging,
    ConfigNode,
    BaseTrainer,
    BaseTrainingPipeline,
    DEFAULT_ARG_SCHEMA,
    format_params,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    coerce_to_str,
    coerce_to_int,
    load_clip_to_cpu,
)

_tokenizer = _Tokenizer()

ARG_SCHEMA = DEFAULT_ARG_SCHEMA


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        n_ctx = compound_prompts_deeper_text[0].shape[0] if len(compound_prompts_deeper_text) > 0 else 0
        
        for i, resblock in enumerate(self.transformer.resblocks):
            if i > 0 and i <= len(compound_prompts_deeper_text):
                prefix = x[:1, :, :]
                suffix = x[1 + n_ctx:, :, :]
                textual_ctx = compound_prompts_deeper_text[i - 1]
                textual_ctx = textual_ctx.expand(x.shape[1], -1, -1).permute(1, 0, 2).to(x.dtype)
                x = torch.cat([prefix, textual_ctx, suffix], dim=0)
            x = resblock(x)
        
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        model_cfg = cfg.get('model', ConfigNode())
        
        n_cls = len(classnames)
        n_ctx = coerce_to_int(model_cfg.get('n_ctx', 2), 2)
        ctx_init = coerce_to_str(model_cfg.get('ctx_init', ''), '')
        prompt_depth = coerce_to_int(model_cfg.get('prompt_depth', 9), 9)
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        assert prompt_depth >= 1, "For MaPLe, prompt_depth should be >= 1"
        self.compound_prompts_depth = prompt_depth

        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        logger.info('MaPLe design: Multi-modal Prompt Learning')
        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of MaPLe context words (tokens): {n_ctx}")
        
        self.proj = nn.Linear(ctx_dim, 768)
        self.ctx = nn.Parameter(ctx_vectors)
        
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, 512))
            for _ in range(self.compound_prompts_depth - 1)
        ])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )
        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts


class VisionEncoder(nn.Module):
    def __init__(self, clip_visual):
        super().__init__()
        self.visual = clip_visual
        self.conv1 = clip_visual.conv1
        self.class_embedding = clip_visual.class_embedding
        self.positional_embedding = clip_visual.positional_embedding
        self.ln_pre = clip_visual.ln_pre
        self.transformer = clip_visual.transformer
        self.ln_post = clip_visual.ln_post
        self.proj = clip_visual.proj

    def forward(self, x, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        
        visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([x, visual_ctx], dim=1)
        n_ctx = shared_ctx.shape[0]
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        
        for i, resblock in enumerate(self.transformer.resblocks):
            if i > 0 and i <= len(compound_deeper_prompts):
                prefix = x[:x.shape[0] - n_ctx, :, :]
                visual_ctx_i = compound_deeper_prompts[i - 1]
                visual_ctx_i = visual_ctx_i.expand(x.shape[1], -1, -1).permute(1, 0, 2).to(x.dtype)
                x = torch.cat([prefix, visual_ctx_i], dim=0)
            x = resblock(x)
        
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        
        return x


class MaPLeCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = VisionEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        
        return logits


class MaPLe(BaseTrainer):
    DEFAULT_LR = 0.0025

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        n_ctx = self._cfg_int(2, 'model.n_ctx')
        prompt_depth = self._cfg_int(9, 'model.prompt_depth')
        
        logger.info(f"Loading CLIP (backbone: {backbone_name})")
        
        clip_model = load_clip_to_cpu(backbone_name)
        
        if self._cfg_str('fp32', 'training.precision', 'precision') in ['fp32', 'amp']:
            clip_model.float()

        self.model = MaPLeCLIP(self.cfg, self.classnames, clip_model)

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"MaPLe: n_ctx={n_ctx}, prompt_depth={prompt_depth}")
        logger.info(f"Learnable parameters: {format_params(trainable_params)} / Total: {format_params(total_params)}")
        
        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}


class MaPLeTrainingPipeline(BaseTrainingPipeline):
    METHOD_NAME = "MaPLe"
    DEFAULT_OUTPUT_DIR = "outputs/maple"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/maple"
    TRAINER_CLASS = MaPLe


def parse_args():
    parser = create_argument_parser("Train MaPLe model", ARG_SCHEMA)
    parsed, unknown = parser.parse_known_args()
    overrides = parse_override_arguments(unknown)
    overrides = process_parsed_args(parsed, ARG_SCHEMA, overrides)
    return parsed, overrides


def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', False))
    base_config = load_config_file(args.config)
    merged = merge_configs(base_config, overrides)
    pipeline = MaPLeTrainingPipeline(merged)
    pipeline.run()


if __name__ == "__main__":
    main()
