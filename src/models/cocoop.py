import torch
import torch.nn as nn
from clip import clip
from collections import OrderedDict
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils import (
    logger,
    ConfigNode,
    BaseTrainer,
    PromptTextEncoder,
    format_params,
    coerce_to_str,
    coerce_to_int,
    load_clip_to_cpu,
)

_tokenizer = _Tokenizer()


class CoCoOPPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        if not isinstance(cfg, ConfigNode):
            cfg = ConfigNode(cfg)
        model_cfg = cfg.get('model', ConfigNode())
        
        n_cls = len(classnames)
        n_ctx = coerce_to_int(model_cfg.get('n_ctx', 4), 4)
        ctx_init = coerce_to_str(model_cfg.get('ctx_init', ''), '')
        
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        bias = self.meta_net(im_features)
        bias = bias.unsqueeze(1)
        ctx = ctx.unsqueeze(0)
        ctx_shifted = ctx + bias

        B = im_features.shape[0]
        n_cls = self.n_cls

        ctx_expanded = ctx_shifted.unsqueeze(1).expand(-1, n_cls, -1, -1)
        prefix_expanded = prefix.unsqueeze(0).expand(B, -1, -1, -1)
        suffix_expanded = suffix.unsqueeze(0).expand(B, -1, -1, -1)

        prompts = torch.cat([prefix_expanded, ctx_expanded, suffix_expanded], dim=2)

        return prompts


class CoCoOPCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CoCoOPPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = PromptTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)

        B, n_cls, L, D = prompts.shape
        prompts_flat = prompts.view(B * n_cls, L, D)
        tokenized_flat = tokenized_prompts.unsqueeze(0).expand(B, -1, -1).reshape(B * n_cls, -1)

        CHUNK_SIZE = 512
        text_features_list = []
        for i in range(0, prompts_flat.shape[0], CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, prompts_flat.shape[0])
            tf_chunk = self.text_encoder(prompts_flat[i:end], tokenized_flat[i:end])
            text_features_list.append(tf_chunk)
        text_features = torch.cat(text_features_list, dim=0)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(B, n_cls, -1)

        logits = logit_scale * torch.einsum('bd,bcd->bc', image_features, text_features)

        return logits


class CoCoOP(BaseTrainer):
    DEFAULT_LR = 0.002

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        logger.info(f"Loading CLIP (backbone: {backbone_name})")
        
        clip_model = load_clip_to_cpu(backbone_name)
        
        if self._cfg_str('fp32', 'training.precision', 'precision') in ['fp32', 'amp']:
            clip_model.float()

        self.model = CoCoOPCLIP(self.cfg, self.classnames, clip_model)

        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        n_ctx = self._cfg_int(4, 'model.n_ctx')
        logger.info(f"CoCoOP: n_ctx={n_ctx}, meta_net enabled")
        logger.info(f"Learnable parameters: {format_params(trainable_params)} / Total: {format_params(total_params)}")
        
        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
