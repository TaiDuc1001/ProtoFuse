import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils import (
    logger,
    ConfigNode,
    BaseTrainer,
    format_params,
    coerce_to_str,
    coerce_to_int,
    load_clip_to_cpu,
    compute_metrics,
)

_tokenizer = _Tokenizer()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaPLeTextEncoder(nn.Module):
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
        
        # logger.info('MaPLe design: Multi-modal Prompt Learning')
        # logger.info(f'Initial context: "{prompt_prefix}"')
        # logger.info(f"Number of MaPLe context words (tokens): {n_ctx}")
        
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
        self.text_encoder = MaPLeTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.fused_prototypes = None

    def encode_image_features(self, image):
        _, shared_ctx, _, deep_compound_prompts_vision = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        return image_features / image_features.norm(dim=-1, keepdim=True)

    def encode_text_features(self):
        prompts, _, deep_compound_prompts_text, _ = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def set_fused_prototypes(self, prototypes):
        self.fused_prototypes = F.normalize(prototypes.detach().to(self.logit_scale.device).float(), dim=-1)

    def clear_fused_prototypes(self):
        self.fused_prototypes = None

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.fused_prototypes is None:
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()
        else:
            prototypes = self.fused_prototypes.to(image_features.device, dtype=image_features.dtype)
            logits = image_features @ prototypes.t()
        
        return logits


class MaPLe(BaseTrainer):
    DEFAULT_LR = 0.0025

    def build_model(self):
        backbone_name = self._cfg_str('ViT-B/16', 'model.backbone', 'backbone')
        n_ctx = self._cfg_int(2, 'model.n_ctx')
        prompt_depth = self._cfg_int(9, 'model.prompt_depth')
        
        # logger.info(f"Loading CLIP (backbone: {backbone_name})")
        
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
        
        # logger.info(f"MaPLe: n_ctx={n_ctx}, prompt_depth={prompt_depth}")
        # logger.info(f"Learnable parameters: {format_params(trainable_params)} / Total: {format_params(total_params)}")
        
        self.model.to(self.device)
        self.initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.text_prototypes = None
        self.visual_centroids = None
        self.fused_prototypes = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def clear_posthoc_protofuse(self):
        self.fused_prototypes = None
        self.visual_centroids = None
        self.posthoc_alpha = None
        self.posthoc_missing_classes = []
        self.model.clear_fused_prototypes()

    def get_text_prototypes(self):
        self.model.eval()
        with torch.no_grad():
            text_features = self.model.encode_text_features().float()
        self.text_prototypes = F.normalize(text_features, dim=-1)
        return self.text_prototypes

    def extract_features(self, dataloader):
        all_features = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                features = self.model.encode_image_features(images).float()
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        if not all_features:
            raise RuntimeError("Cannot extract MaPLe features from an empty dataloader.")
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def build_visual_centroids(self, features, labels, num_classes=None, text_prototypes=None):
        if features.numel() == 0 or labels.numel() == 0:
            raise RuntimeError("Cannot build visual centroids without training samples.")

        num_classes = len(self.classnames) if num_classes is None else num_classes
        features = F.normalize(features.to(self.device).float(), dim=-1)
        labels = labels.to(self.device).long()

        centroids = torch.zeros(num_classes, features.shape[-1], device=self.device, dtype=features.dtype)
        counts = torch.zeros(num_classes, device=self.device, dtype=features.dtype)
        valid_mask = (labels >= 0) & (labels < num_classes)
        if valid_mask.any():
            centroids.index_add_(0, labels[valid_mask], features[valid_mask])
            counts.index_add_(0, labels[valid_mask], torch.ones_like(labels[valid_mask], dtype=features.dtype))

        present = counts > 0
        if present.any():
            centroids[present] = F.normalize(centroids[present] / counts[present].unsqueeze(1), dim=-1)

        if text_prototypes is not None:
            text_prototypes = F.normalize(text_prototypes.to(self.device).float(), dim=-1)
            centroids[~present] = text_prototypes[~present]

        self.visual_centroids = centroids
        self.posthoc_missing_classes = torch.nonzero(~present, as_tuple=False).flatten().cpu().tolist()
        return centroids

    def apply_posthoc_protofuse(self, alpha, fused_prototypes, visual_centroids=None, missing_classes=None):
        self.freeze()
        alpha = max(0.0, min(1.0, alpha))
        self.fused_prototypes = F.normalize(fused_prototypes.to(self.device).float(), dim=-1)
        self.visual_centroids = visual_centroids.to(self.device) if visual_centroids is not None else None
        self.posthoc_alpha = alpha
        self.posthoc_missing_classes = list(missing_classes or [])
        self.model.set_fused_prototypes(self.fused_prototypes)
        return self.fused_prototypes

    def logits_from_features(self, features, prototypes=None):
        image_features = F.normalize(features.to(self.device).float(), dim=-1)
        if prototypes is None:
            prototypes = self.fused_prototypes if self.fused_prototypes is not None else self.get_text_prototypes()
        prototypes = F.normalize(prototypes.to(self.device).float(), dim=-1)
        return image_features @ prototypes.t()

    def evaluate_features(self, features, labels, prototypes=None, alpha=None):
        labels_device = labels.to(self.device).long()
        with torch.no_grad():
            logits = self.logits_from_features(features, prototypes=prototypes)
            loss = F.cross_entropy(logits, labels_device)
            preds = logits.argmax(dim=-1)

        labels_list = labels_device.cpu().tolist()
        preds_list = preds.cpu().tolist()
        metrics = compute_metrics(labels_list, preds_list)
        metrics['loss'] = loss.item()
        metrics['predictions'] = preds_list
        metrics['true_labels'] = labels_list
        if alpha is not None:
            metrics['alpha'] = alpha
        return metrics

    def save_posthoc_protofuse(self, path):
        torch.save({
            'text_prototypes': self.text_prototypes.detach().cpu() if self.text_prototypes is not None else None,
            'visual_centroids': self.visual_centroids.detach().cpu() if self.visual_centroids is not None else None,
            'fused_prototypes': self.fused_prototypes.detach().cpu() if self.fused_prototypes is not None else None,
            'alpha': self.posthoc_alpha,
            'missing_classes': self.posthoc_missing_classes,
            'classnames': self.classnames,
        }, path)

    def save_model(self, path):
        checkpoint = {
            'prompt_learner_state_dict': self.model.prompt_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'cfg': self.cfg,
            'posthoc_protofuse': {
                'text_prototypes': self.text_prototypes.detach().cpu() if self.text_prototypes is not None else None,
                'visual_centroids': self.visual_centroids.detach().cpu() if self.visual_centroids is not None else None,
                'fused_prototypes': self.fused_prototypes.detach().cpu() if self.fused_prototypes is not None else None,
                'alpha': self.posthoc_alpha,
                'missing_classes': self.posthoc_missing_classes,
            },
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint['prompt_learner_state_dict']
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]
        self.model.prompt_learner.load_state_dict(state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        posthoc = checkpoint.get('posthoc_protofuse') or {}
        fused = posthoc.get('fused_prototypes')
        self.text_prototypes = posthoc.get('text_prototypes')
        self.visual_centroids = posthoc.get('visual_centroids')
        self.posthoc_alpha = posthoc.get('alpha')
        self.posthoc_missing_classes = posthoc.get('missing_classes') or []
        if self.text_prototypes is not None:
            self.text_prototypes = self.text_prototypes.to(self.device)
        if self.visual_centroids is not None:
            self.visual_centroids = self.visual_centroids.to(self.device)
        if fused is not None:
            self.fused_prototypes = fused.to(self.device)
            self.model.set_fused_prototypes(self.fused_prototypes)
        else:
            self.fused_prototypes = None
            self.model.clear_fused_prototypes()
