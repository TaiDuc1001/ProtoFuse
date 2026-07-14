from utils import (
    ConfigNode,
    coerce_to_float,
    coerce_to_int,
    get_config_value,
    load_config_file,
    logger,
)

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
        rho = None
        return alpha_steps, beta_values, rho
