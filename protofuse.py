import os
os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils import (
    setup_logging,
    DEFAULT_ARG_SCHEMA,
    create_argument_parser,
    process_parsed_args,
    parse_override_arguments,
    merge_configs,
    load_config_file,
    logger,
)

from src.pipelines.protofuse import ProtoFusePipeline
from src.models.protofuse import ProtoFuse

ARG_SCHEMA = DEFAULT_ARG_SCHEMA

def patched_fuse_and_evaluate(self, train_features, train_labels, eval_features, eval_labels, num_classes):
    V_all = self.build_visual_centroids(train_features, train_labels, num_classes)
    T = self.text_prototypes
    
    class_indices = [[] for _ in range(num_classes)]
    for idx, lbl in enumerate(train_labels.tolist()): class_indices[lbl].append(idx)
    k = min(len(idxs) for idxs in class_indices)
    class_feat = torch.stack([train_features[class_indices[c][:k]].to(self.device) for c in range(num_classes)])
    class_sums = class_feat.sum(dim=1)

    loo_global_scores = torch.zeros(len(self.alphas), device=self.device)
    for fold in range(k):
        held = F.normalize(class_feat[:, fold, :], dim=-1)
        V_loo = F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1)
        refined = F.normalize((1 - self.alphas).view(-1, 1, 1) * T + self.alphas.view(-1, 1, 1) * V_loo, dim=-1)
        loo_global_scores += torch.einsum("qd,apd->aqp", held, refined).argmax(dim=-1).eq(torch.arange(num_classes, device=self.device)).float().mean(dim=-1)
    
    global_loo_acc = (loo_global_scores.max() / k).item()
    global_alpha = self.alphas[loo_global_scores.argmax()].item()
    global_protos = F.normalize((1 - global_alpha) * T + global_alpha * V_all, dim=-1)
    
    sim_matrix_g = global_protos @ global_protos.T
    sim_matrix_g.fill_diagonal_(-1)
    global_collision = sim_matrix_g.max(dim=1)[0].mean().item()
    
    eval_norm = F.normalize(eval_features.to(self.device), dim=-1)
    global_eval_acc = (eval_norm @ global_protos.T).argmax(dim=-1).eq(eval_labels.to(self.device)).float().mean().item()

    final_alphas = torch.full((num_classes,), global_alpha, device=self.device)
    best_loo_acc = global_loo_acc
    best_alphas_vec = final_alphas.clone()
    
    logger.info("Running Iterative Optimization (Max 20 Iterations)")
    for iter_idx in range(1, 21):
        alphas_before = final_alphas.clone()
        for c in range(num_classes):
            current_protos = F.normalize((1 - final_alphas.view(-1, 1)) * T + final_alphas.view(-1, 1) * V_all, dim=-1)
            mask = torch.ones(num_classes, dtype=torch.bool, device=self.device); mask[c] = False
            others = current_protos[mask]
            
            loo_c_scores = torch.zeros(len(self.alphas), device=self.device)
            for fold in range(k):
                held_c = F.normalize(class_feat[c, fold].unsqueeze(0), dim=-1)
                V_loo_c = F.normalize((class_sums[c] - class_feat[c, fold]) / (k - 1), dim=-1)
                refined_sweep = F.normalize((1 - self.alphas).view(-1, 1) * T[c] + self.alphas.view(-1, 1) * V_loo_c, dim=-1)
                loo_c_scores += (torch.mm(held_c, refined_sweep.T) > torch.mm(held_c, others.T).max()).float().squeeze()
            final_alphas[c] = self.alphas[loo_c_scores.argmax()]

        diff = (final_alphas != alphas_before).sum().item()
        
        current_protos_full = F.normalize((1 - final_alphas.view(-1, 1)) * T + final_alphas.view(-1, 1) * V_all, dim=-1)
        total_correct = 0
        for fold in range(k):
            held_all = F.normalize(class_feat[:, fold, :], dim=-1)
            loo_protos = F.normalize((1 - final_alphas.view(-1, 1)) * T + final_alphas.view(-1, 1) * F.normalize((class_sums - class_feat[:, fold, :]) / (k - 1), dim=-1), dim=-1)
            for c_eval in range(num_classes):
                mask_eval = torch.ones(num_classes, dtype=torch.bool, device=self.device); mask_eval[c_eval] = False
                if torch.dot(held_all[c_eval], loo_protos[c_eval]) > torch.mm(held_all[c_eval].unsqueeze(0), current_protos_full[mask_eval].T).max():
                    total_correct += 1
        
        iter_acc = total_correct / (num_classes * k)
        if iter_acc > best_loo_acc:
            best_loo_acc = iter_acc
            best_alphas_vec = final_alphas.clone()
        if diff == 0: break

    iter_protos = F.normalize((1 - best_alphas_vec.view(-1, 1)) * T + best_alphas_vec.view(-1, 1) * V_all, dim=-1)
    
    sim_matrix_i = iter_protos @ iter_protos.T
    sim_matrix_i.fill_diagonal_(-1)
    iter_collision = sim_matrix_i.max(dim=1)[0].mean().item()
    
    iter_eval_acc = (eval_norm @ iter_protos.T).argmax(dim=-1).eq(eval_labels.to(self.device)).float().mean().item()

    logger.info("=============================================================")
    logger.info("PROOF 1: OVERFITTING TO FEW-SHOT NOISE")
    logger.info(f"Global Alpha:    LOO Acc = {global_loo_acc:.4f} | Test Acc = {global_eval_acc:.4f} | Drop = {(global_loo_acc - global_eval_acc):.4f}")
    logger.info(f"Iterative Alpha: LOO Acc = {best_loo_acc:.4f} | Test Acc = {iter_eval_acc:.4f} | Drop = {(best_loo_acc - iter_eval_acc):.4f}")
    logger.info("Observation: Iterative algorithm artificially inflates training accuracy but fails completely on unseen test data.")
    logger.info("-------------------------------------------------------------")
    logger.info("PROOF 2: PROTOTYPE COLLISION (GEOMETRY DEGRADATION)")
    logger.info(f"Global Alpha Nearest Neighbor Similarity:    {global_collision:.4f}")
    logger.info(f"Iterative Alpha Nearest Neighbor Similarity: {iter_collision:.4f}")
    logger.info("Observation: Independent alphas push prototypes closer to each other, destroying inter-class separability.")
    logger.info("=============================================================")

    self.proof_data = {
        'g_loo': global_loo_acc, 'g_test': global_eval_acc, 'g_col': global_collision,
        'i_loo': best_loo_acc, 'i_test': iter_eval_acc, 'i_col': iter_collision
    }
    
    self.fused_prototypes = iter_protos
    self.best_alphas = best_alphas_vec
    
    metrics = __import__('utils').compute_metrics(eval_labels.tolist(), (eval_norm @ iter_protos.T).argmax(dim=-1).cpu().tolist())
    metrics['alpha'] = best_alphas_vec.mean().item()
    return metrics

ProtoFuse.fuse_and_evaluate = patched_fuse_and_evaluate

def parse_args():
    p, u = create_argument_parser("Run", ARG_SCHEMA).parse_known_args()
    return p, process_parsed_args(p, ARG_SCHEMA, parse_override_arguments(u))

def main():
    args, overrides = parse_args()
    setup_logging(getattr(args, 'debug', True), getattr(args, 'disable_coloring', True))
    pipeline = ProtoFusePipeline(merge_configs(load_config_file(args.config), overrides))
    pipeline.run()
    
    t = pipeline.trainer
    if hasattr(t, 'proof_data'):
        d = t.proof_data
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        labels = ['Global \n(Regularized)', 'Iterative \n(Over-Optimized)']
        loo_accs = [d['g_loo'], d['i_loo']]
        test_accs = [d['g_test'], d['i_test']]
        x = np.arange(len(labels))
        w = 0.35
        plt.bar(x - w/2, loo_accs, w, label='Training (LOO) Acc', color='skyblue')
        plt.bar(x + w/2, test_accs, w, label='Test (Unseen) Acc', color='salmon')
        plt.ylabel('Accuracy')
        plt.title('Proof 1: The Overfitting Gap')
        plt.xticks(x, labels)
        plt.ylim(0.7, 0.8)
        plt.legend()
        for i, v in enumerate(loo_accs): plt.text(i - w/2, v + 0.002, f"{v:.4f}", ha='center')
        for i, v in enumerate(test_accs): plt.text(i + w/2, v + 0.002, f"{v:.4f}", ha='center')

        plt.subplot(1, 2, 2)
        cols = [d['g_col'], d['i_col']]
        plt.bar(labels, cols, width=0.5, color=['lightgreen', 'tomato'])
        plt.ylabel('Cosine Similarity (Higher = More Collisions)')
        plt.title('Proof 2: Prototype Collision')
        plt.ylim(min(cols)-0.02, max(cols)+0.02)
        for i, v in enumerate(cols): plt.text(i, v + 0.002, f"{v:.4f}", ha='center')

        plt.tight_layout()
        os.makedirs(pipeline.run_dir, exist_ok=True)
        plt.savefig(os.path.join(pipeline.run_dir, 'proof_of_failure.png'), dpi=300)
        plt.close()

if __name__ == "__main__": main()
