import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
from thop import profile

from utils import ConfigNode, load_clip_to_cpu, load_config_file
from coop import CoOPCLIP
from maple import MaPLeCLIP
from apt import CustomCLIP as APTCLIP
from vife import TransformerAdapter, SSLHead, LinearClassifier, FusionWeightLearner


CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
NUM_CLASSES = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITERS = 10
BENCHMARK_ITERS = 100


def load_config(config_name):
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config_file(config_path)
    return ConfigNode(config)


def count_params(module, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_params(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(int(num))


def get_gflops(model, input_size=(1, 3, 224, 224)):
    dummy_input = torch.randn(input_size).to(DEVICE)
    model = model.to(DEVICE)
    model.eval()
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return macs / 1e9
    except Exception as e:
        print(f"  [Warning] GFLOPs computation failed: {e}")
        return None


def benchmark_fps(model_builder, input_size=(1, 3, 224, 224), warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    model = model_builder()
    dummy_input = torch.randn(input_size).to(DEVICE)
    model = model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        for _ in range(iters):
            _ = model(dummy_input)
        
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start_time
    
    fps = iters / elapsed
    latency_ms = (elapsed / iters) * 1000
    return fps, latency_ms


def analyze_coop(clip_model, classnames):
    cfg = load_config("coop")
    print(f"  Config: n_ctx={cfg.model.n_ctx}, csc={cfg.model.get('csc', False)}")
    
    model = CoOPCLIP(cfg, classnames, clip_model)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    learnable = count_params(model, only_trainable=True)
    
    details = {}
    for name, param in model.prompt_learner.named_parameters():
        if param.requires_grad:
            details[name] = param.numel()
    
    return model, learnable, details, cfg


def analyze_maple(clip_model, classnames):
    cfg = load_config("maple")
    print(f"  Config: n_ctx={cfg.model.n_ctx}, prompt_depth={cfg.model.prompt_depth}")
    
    model = MaPLeCLIP(cfg, classnames, clip_model)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    learnable = count_params(model, only_trainable=True)
    
    details = {}
    for name, param in model.prompt_learner.named_parameters():
        if param.requires_grad:
            details[name] = param.numel()
    
    return model, learnable, details, cfg


def analyze_apt(clip_model, classnames):
    cfg = load_config("apt")
    print(f"  Config: num_heads={cfg.model.num_heads}, num_layers={cfg.model.num_layers}, dropout={cfg.model.dropout}")
    
    model = APTCLIP(cfg, classnames, clip_model, DEVICE)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    learnable = count_params(model, only_trainable=True)
    
    details = {}
    for name, param in model.prompt_learner.named_parameters():
        if param.requires_grad:
            details[name] = param.numel()
    
    return model, learnable, details, cfg


def get_module_gflops(module, input_shape):
    dummy_input = torch.randn(input_shape).to(DEVICE)
    module = module.to(DEVICE)
    module.eval()
    try:
        with torch.no_grad():
            macs, _ = profile(module, inputs=(dummy_input,), verbose=False)
        return macs / 1e9
    except Exception:
        return 0.0


def analyze_vife(clip_model, classnames):
    cfg = load_config("vife")
    ssl_cfg = cfg.get("ssl", ConfigNode())
    
    feature_dim = 768
    proj_dim = ssl_cfg.get("proj_dim", 256)
    num_prototypes = ssl_cfg.get("num_prototypes", 4096)
    num_trans_layers = ssl_cfg.get("num_trans_layers", 1)
    num_heads = ssl_cfg.get("num_heads", 8)
    
    print(f"  Config: proj_dim={proj_dim}, num_prototypes={num_prototypes}, trans_layers={num_trans_layers}")

    apt_model, apt_params, apt_details, _ = analyze_apt.__wrapped__(clip_model, classnames) if hasattr(analyze_apt, '__wrapped__') else _analyze_apt_for_vife(clip_model, classnames, cfg)

    adapter = TransformerAdapter(feature_dim, num_trans_layers, num_heads)
    ssl_head = SSLHead(feature_dim, proj_dim, num_prototypes)
    classifier = LinearClassifier(feature_dim, NUM_CLASSES)
    fusion = FusionWeightLearner()

    adapter_params = count_params(adapter, only_trainable=True)
    ssl_head_params = count_params(ssl_head, only_trainable=True)
    classifier_params = count_params(classifier, only_trainable=True)
    fusion_params = count_params(fusion, only_trainable=True)

    total = apt_params + adapter_params + ssl_head_params + classifier_params + fusion_params

    adapter_gflops = get_module_gflops(adapter, (1, 197, feature_dim))
    ssl_head_gflops = get_module_gflops(ssl_head, (1, feature_dim))
    classifier_gflops = get_module_gflops(classifier, (1, feature_dim))

    ssl_total_gflops = adapter_gflops + ssl_head_gflops + classifier_gflops

    details = {
        'APT (CrossAttention)': apt_params,
        'TransformerAdapter': adapter_params,
        'SSLHead': ssl_head_params,
        'LinearClassifier': classifier_params,
        'FusionWeightLearner': fusion_params
    }

    gflops_details = {
        'TransformerAdapter': adapter_gflops,
        'SSLHead': ssl_head_gflops,
        'LinearClassifier': classifier_gflops,
        'SSL Total': ssl_total_gflops
    }

    return apt_model, total, details, cfg, gflops_details


def _analyze_apt_for_vife(clip_model, classnames, vife_cfg):
    model = APTCLIP(vife_cfg, classnames, clip_model, DEVICE)
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    
    learnable = count_params(model, only_trainable=True)
    
    details = {}
    for name, param in model.prompt_learner.named_parameters():
        if param.requires_grad:
            details[name] = param.numel()
    
    return model, learnable, details, vife_cfg


def main():
    print("=" * 70)
    print("LEARNABLE PARAMETERS vs GFLOPs COMPARISON")
    print("=" * 70)
    print(f"\nConfigs directory: {CONFIGS_DIR}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Device: {DEVICE}\n")

    coop_cfg = load_config("coop")
    backbone = coop_cfg.model.backbone
    print(f"Backbone: {backbone}")
    print("Loading CLIP model...")
    clip_model = load_clip_to_cpu(backbone)
    clip_model.float()

    classnames = [f"class_{i}" for i in range(NUM_CLASSES)]

    print("\n" + "-" * 70)
    print("ANALYZING EACH METHOD...")
    print("-" * 70)

    print("\n[1/4] CoOp (Context Optimization)")
    coop_model, coop_params, coop_details, coop_cfg = analyze_coop(clip_model, classnames)
    coop_gflops = get_gflops(coop_model)
    print(f"  Learnable components:")
    for name, num in coop_details.items():
        print(f"    - {name}: {format_params(num)} ({num:,})")
    print(f"  TOTAL: {format_params(coop_params)}")

    print("\n[2/4] MaPLe (Multi-modal Prompt Learning)")
    maple_model, maple_params, maple_details, maple_cfg = analyze_maple(clip_model, classnames)
    maple_gflops = get_gflops(maple_model)
    print(f"  Learnable components:")
    for name, num in maple_details.items():
        print(f"    - {name}: {format_params(num)} ({num:,})")
    print(f"  TOTAL: {format_params(maple_params)}")

    print("\n[3/4] APT (Adapted Prompt Tuning)")
    apt_model, apt_params, apt_details, apt_cfg = analyze_apt(clip_model, classnames)
    apt_gflops = get_gflops(apt_model)
    print(f"  Learnable components:")
    for name, num in apt_details.items():
        print(f"    - {name}: {format_params(num)} ({num:,})")
    print(f"  TOTAL: {format_params(apt_params)}")

    print("\n[4/4] ViFE (Visual Finegrained Extractor = APT + SSL)")
    vife_model, vife_params, vife_details, vife_cfg, vife_gflops_details = analyze_vife(clip_model, classnames)
    vife_gflops = apt_gflops + vife_gflops_details['SSL Total']
    print(f"  Learnable components:")
    for name, num in vife_details.items():
        print(f"    - {name}: {format_params(num)} ({num:,})")
    print(f"  TOTAL: {format_params(vife_params)}")
    print(f"  SSL GFLOPs breakdown:")
    print(f"    - APT (base): {apt_gflops:.4f}")
    for name, gf in vife_gflops_details.items():
        if name != 'SSL Total':
            print(f"    - {name}: {gf:.4f}")
    print(f"    - TOTAL: {vife_gflops:.4f}")

    print("\n" + "=" * 70)
    print("BENCHMARKING FPS...")
    print("=" * 70)
    print(f"Warmup: {WARMUP_ITERS} iterations, Benchmark: {BENCHMARK_ITERS} iterations")
    
    print("\nReloading CLIP model for clean FPS benchmark...")
    clip_model_fps = load_clip_to_cpu(backbone)
    clip_model_fps.float()
    
    print("Measuring CoOp...")
    coop_fps, coop_latency = benchmark_fps(lambda: CoOPCLIP(coop_cfg, classnames, clip_model_fps))
    print(f"  FPS: {coop_fps:.2f}, Latency: {coop_latency:.2f}ms")
    
    print("Measuring MaPLe...")
    maple_fps, maple_latency = benchmark_fps(lambda: MaPLeCLIP(maple_cfg, classnames, clip_model_fps))
    print(f"  FPS: {maple_fps:.2f}, Latency: {maple_latency:.2f}ms")
    
    print("Measuring APT...")
    apt_fps, apt_latency = benchmark_fps(lambda: APTCLIP(apt_cfg, classnames, clip_model_fps, DEVICE))
    print(f"  FPS: {apt_fps:.2f}, Latency: {apt_latency:.2f}ms")
    
    print("Measuring ViFE (APT inference)...")
    vife_fps, vife_latency = benchmark_fps(lambda: APTCLIP(vife_cfg, classnames, clip_model_fps, DEVICE))
    print(f"  FPS: {vife_fps:.2f}, Latency: {vife_latency:.2f}ms")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<12} {'Params':>12} {'GFLOPs':>10} {'FPS':>10} {'Latency':>12}")
    print("-" * 70)
    
    results = [
        ("CoOp", coop_params, coop_gflops, coop_fps, coop_latency),
        ("MaPLe", maple_params, maple_gflops, maple_fps, maple_latency),
        ("APT", apt_params, apt_gflops, apt_fps, apt_latency),
        ("ViFE", vife_params, vife_gflops, vife_fps, vife_latency),
    ]
    
    for method, params, gflops, fps, latency in results:
        params_str = format_params(params)
        gflops_str = f"{gflops:.2f}" if gflops else "N/A"
        print(f"{method:<12} {params_str:>12} {gflops_str:>10} {fps:>10.2f} {latency:>10.2f}ms")
    
    print("=" * 70)
    
    print("\nCONFIG SETTINGS USED:")
    print(f"- CoOp: n_ctx={coop_cfg.model.n_ctx}")
    print(f"- MaPLe: n_ctx={maple_cfg.model.n_ctx}, prompt_depth={maple_cfg.model.prompt_depth}")
    print(f"- APT: num_heads={apt_cfg.model.num_heads}, num_layers={apt_cfg.model.num_layers}")
    print(f"- ViFE: proj_dim={vife_cfg.ssl.proj_dim}, num_prototypes={vife_cfg.ssl.num_prototypes}")
    
    print(f"\nNOTE: FPS measured on {DEVICE.upper()} with batch_size=1")


if __name__ == "__main__":
    main()
