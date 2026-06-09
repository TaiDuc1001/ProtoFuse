from utils import (
    ConfigNode,
    coerce_to_int,
    get_config_value,
    load_config_file,
    logger,
)


def coerce_protofuse_bool(raw, default=False):
    if raw is None:
        return bool(default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if value in {'0', 'false', 'no', 'n', 'off'}:
            return False
        return bool(default)
    return bool(raw)


def resolve_force_loo_accuracy(cfg, proto_cfg):
    proto_default = get_config_value(
        proto_cfg,
        'model.force_loo_accuracy',
        get_config_value(proto_cfg, 'force_loo_accuracy', False),
    )
    return coerce_protofuse_bool(cfg.get('force_loo_accuracy', proto_default), False)


def resolve_force_weighted_centroid(cfg, proto_cfg):
    proto_default = get_config_value(
        proto_cfg,
        'model.force_weighted_centroid',
        get_config_value(proto_cfg, 'force_weighted_centroid', True),
    )
    return coerce_protofuse_bool(cfg.get('force_weighted_centroid', proto_default), True)


class PosthocProtoFuseMixin:
    def _posthoc_protofuse_cfg(self):
        return self.config.get('posthoc_protofuse', ConfigNode())

    def _posthoc_protofuse_enabled(self):
        return bool(self._posthoc_protofuse_cfg().get('enabled', False))

    def _posthoc_protofuse_selector_settings(self):
        cfg = self._posthoc_protofuse_cfg()
        proto_cfg = {}
        config_path = cfg.get('config_path', 'configs/protofuse.yaml')
        if config_path:
            try:
                proto_cfg = load_config_file(config_path)
            except FileNotFoundError:
                logger.warning(f"ProtoFuse config not found at {config_path}; using post-hoc defaults.")

        alpha_steps = coerce_to_int(
            cfg.get('alpha_steps', get_config_value(proto_cfg, 'model.alpha_steps', 101)),
            101,
            key='posthoc_protofuse.alpha_steps',
        )
        proto_beta_values = get_config_value(proto_cfg, 'model.centroid_mix.beta_values', None)
        centroid_mix_cfg = cfg.get('centroid_mix', ConfigNode())
        beta_values = centroid_mix_cfg.get('beta_values', proto_beta_values)
        force_loo_accuracy = resolve_force_loo_accuracy(cfg, proto_cfg)
        force_weighted_centroid = resolve_force_weighted_centroid(cfg, proto_cfg)
        return alpha_steps, beta_values, force_loo_accuracy, force_weighted_centroid
