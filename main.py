"""
main.py — RAPID v3: Block-wise Absolute WTA + Prototype Classifier (NCM)
==========================================================================
Cải tiến so với phiên bản cũ:
  [YC1] Block-wise WTA với Absolute Top-k (Paper Eq.4 / Theorem 4.2)
  [YC2] Prototype Classifier (NCM + Cosine Similarity) thay Ridge Regression
  [YC3] Data normalization [-1,1] cho ViT-B/16 (Paper Appendix C.3)
        → Truyền --data_augmentation vit khi chạy (default đã set).
"""
import argparse
import time
from typing import Optional

import torch
import numpy as np
from torch.nn import functional as F
from sklearn.cluster import KMeans

from datasets.load_dataset import load_dataset
from models.load_model import load_model
from utils import random_initialization, feature_extract


# =============================================================================
# Argument Parser
# =============================================================================
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAPID v3: Block-wise Absolute WTA + Prototype Classifier"
    )
    # CIL task setting
    parser.add_argument('--dataset',     default='CIFAR-100', help='Choose dataset')
    parser.add_argument('--root',        default='../data',   help='Dataset root path')
    parser.add_argument('--num_classes', type=int, default=100, help='Total number of classes')
    parser.add_argument('--num_tasks',   type=int, default=10,  help='Number of tasks')

    # Model / Projection
    parser.add_argument('--model_name',      default="vit_base_patch16_224",
                        help='Backbone model name')
    parser.add_argument('--embedding_dim',   type=int, default=768,
                        help='Backbone output dimension')
    parser.add_argument('--expand_dim',      type=int, default=10000,
                        help='Total neuron budget (split across tasks)')
    parser.add_argument('--synaptic_degree', type=int, default=100,
                        help='Non-zero connections per row p')
    # [YC1] coding_level áp dụng PER BLOCK: k = coding_level × block_size
    parser.add_argument('--coding_level',    type=float, default=0.01,
                        help='Top-k ratio per block (default=0.01 for better sparsity)')


    parser.add_argument('--classifier_type', type=str, default='ncm',
                        choices=['ncm', 'ridge_subspace', 'ridge'],
                        help='Type of classifier: ncm, ridge_subspace, ridge')
    parser.add_argument('--use_subspace', action='store_true', default=False,
                        help='Enable subspace extraction for ridge_subspace')
    parser.add_argument('--use_acp', action='store_true', default=False,
                        help='Enable Analytic Contrastive Projection on subspace')
    parser.add_argument('--subspace_rank', type=int, default=50,
                        help='Rank r of the subspace (default 50)')
    parser.add_argument('--acp_alpha', type=float, default=1.0,
                        help='Alpha parameter for ACP separation enhancement')
    # [ETF] Bật/Tắt căn chỉnh Procrustes
    parser.add_argument('--use_procrustes', action='store_true',
                        help='Sử dụng ETF Prototypes và Procrustes Alignment')
    
    # [YC2] K prototypes per class
    parser.add_argument('--num_prototypes', type=int, default=3,
                        help='Number of prototypes per class (K-Means)')

    # Fisher-adaptive allocation
    parser.add_argument('--fisher_block', type=int,   default=512,
                        help='Step size for Fisher probe (neurons per step)')
    parser.add_argument('--fisher_sat',   type=float, default=0.005,
                        help='Fisher saturation threshold δ')

    # [YC3] Normalization — "vit" → [-1,1] theo Paper Appendix C.3
    parser.add_argument('--data_augmentation', default='vit',
                        choices=[None, 'resnet', 'vit'],
                        help='"vit"→Normalize([0.5],[0.5])=[-1,1]; '
                             '"resnet"→ImageNet; None→raw [0,1]')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed',       type=int, default=2025, help='Random seed')
    parser.add_argument('--gpu',        type=int, default=0,    help='GPU index')
    return parser


# =============================================================================
# Sparse Random Projection Block
# =============================================================================
def _make_sparse_projection_block(
    num_neurons: int,
    embedding_dim: int,
    non_zero_per_row: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Tạo sparse random projection block (num_neurons × embedding_dim).
    Mỗi hàng có đúng non_zero_per_row kết nối != 0, lấy từ N(0,1).
    """
    W = torch.zeros(num_neurons, embedding_dim)
    for row in range(num_neurons):
        cols = torch.randperm(embedding_dim)[:non_zero_per_row]
        W[row, cols] = torch.randn(non_zero_per_row)
    return W.to(device)


# =============================================================================
# Fisher Score — phát hiện saturation để dừng grow neuron
# =============================================================================
def compute_mean_fisher_score(
    H: torch.Tensor,        # (N, m) projected features
    labels: torch.Tensor,   # (N,) integer labels
) -> float:
    """
    Fisher Score trung bình = E_neuron[ S_B / (S_B + S_W) ] ∈ [0,1].
    Đo khả năng phân tách class trong không gian chiếu hiện tại.
    """
    N, D = H.shape
    overall_mean = H.mean(dim=0)
    S_B = torch.zeros(D, device=H.device)
    S_W = torch.zeros(D, device=H.device)
    for c in labels.unique():
        mask = (labels == c)
        H_c  = H[mask]
        mu_c = H_c.mean(dim=0)
        S_B += mask.sum() * (mu_c - overall_mean).pow(2)
        S_W += (H_c - mu_c).pow(2).sum(dim=0)
    return (S_B / (S_B + S_W + 1e-12)).mean().item()


def adaptive_projection_size(
    train_embs:  torch.Tensor,
    train_labels: torch.Tensor,
    embedding_dim: int,
    synaptic_degree: int,
    max_neurons: int,
    block_size: int   = 512,
    saturation_threshold: float = 0.005,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Tự động xác định số neuron cần thiết bằng Fisher saturation.

    Bắt đầu với block_size neuron; cứ thêm block_size và đo Δ Fisher Score.
    Dừng khi Δ < saturation_threshold hoặc đạt max_neurons.

    Returns:
        dense Tensor (adaptive_m, embedding_dim)
    """
    proj       = _make_sparse_projection_block(block_size, embedding_dim, synaptic_degree, device)
    H          = train_embs @ proj.T
    prev_score = compute_mean_fisher_score(H, train_labels.to(device))
    print(f"    [Fisher] neurons={proj.shape[0]:5d}  score={prev_score:.4f}")

    while proj.shape[0] < max_neurons:
        remaining = max_neurons - proj.shape[0]
        add_n     = min(block_size, remaining)
        new_blk   = _make_sparse_projection_block(add_n, embedding_dim, synaptic_degree, device)
        proj      = torch.cat([proj, new_blk], dim=0)
        del new_blk

        H_new = train_embs @ proj.T
        score = compute_mean_fisher_score(H_new, train_labels.to(device))
        delta = score - prev_score
        print(f"    [Fisher] neurons={proj.shape[0]:5d}  score={score:.4f}  Δ={delta:+.4f}")
        del H_new

        if prev_score > 0 and delta < saturation_threshold:
            print(f"    [Fisher] Saturation → stop at {proj.shape[0]} neurons.")
            break
        prev_score = score

    return proj


# =============================================================================
# [YC1] Block-wise WTA với Absolute Top-k  (Paper Eq.4 / Theorem 4.2)
# =============================================================================
def blockwise_wta(
    H_full: torch.Tensor,      # (D_total, N) — projected features
    block_sizes: list[int],    # [m_0, m_1, ...] kích thước từng block theo task
    coding_level: float,       # k/block_size — áp dụng riêng cho mỗi block
) -> torch.Tensor:
    """
    [YC1] WTA độc lập trong từng block dùng Absolute Top-k.

    Lý do dùng |x|.topk thay vì x.topk(largest=True):
      • Paper Eq.4: h'_i = x_i nếu |x_i| thuộc top-k, ngược lại = 0.
      • WTA toàn cục với largest=True bỏ qua activation âm lớn (ức chế),
        làm mất thông tin và gây thiên lệch về kích hoạt dương.
      • WTA per-block bảo vệ neuron task cũ khỏi bị "chết đói".
      • Theorem 4.2: tổng k ∝ m → sai số hội tụ về 0.

    Returns:
        H_wta : (D_total, N) — sparse feature với per-block absolute WTA.
    """
    parts  = []
    offset = 0
    for bs in block_sizes:
        blk = H_full[offset: offset + bs, :]          # (bs, N)
        k   = max(1, int(bs * coding_level))

        # [YC1 CORE] Top-k theo độ lớn |x| (không phải largest positive)
        _, topk_idx = torch.abs(blk).topk(k, dim=0)   # (k, N) — index
        # Lấy giá trị thực (giữ nguyên dấu âm/dương) tại các index đó
        topk_vals   = torch.gather(blk, dim=0, index=topk_idx)  # (k, N)

        blk_sparse  = torch.zeros_like(blk)
        blk_sparse.scatter_(0, topk_idx, topk_vals)   # (bs, N)
        parts.append(blk_sparse)
        offset += bs

    return torch.cat(parts, dim=0)                    # (D_total, N)


# =============================================================================
# Helper: ETF Prototypes & Procrustes Alignment
# =============================================================================
def create_etf_prototypes(num_classes: int, feat_dim: int) -> torch.Tensor:
    """
    Tạo ETF chuẩn (C, D) với thuộc tính:
      - ||p_i|| = 1 với mọi i
      - p_i · p_j = -1/(C-1) với mọi i≠j  (equiangular)
    Dùng công thức: P = sqrt(C/(C-1)) * H @ U_base
    với H = I - (1/C)*11^T là centering matrix,
         U_base là ma trận (C, D) với các hàng trực giao.
    """
    if num_classes == 1:
        return F.normalize(torch.randn(1, feat_dim), p=2, dim=1).cpu()

    C, D = num_classes, feat_dim

    # Bước 1: Tạo ma trận U_base (C, D) với các hàng/cột trực chuẩn
    if D >= C:
        rand_mat = torch.randn(D, C)
        Q, _ = torch.linalg.qr(rand_mat)          # Q: (D, C)
        U_base = Q.T                              # (C, D)
    else:
        rand_mat = torch.randn(C, C)
        Q, _ = torch.linalg.qr(rand_mat)          # Q: (C, C)
        U_base = Q[:, :D]                         # (C, D)

    # Bước 2: Áp dụng Centering Matrix H để tạo ETF
    ones  = torch.ones(C, 1)
    H_ctr = torch.eye(C) - (1.0 / C) * (ones @ ones.T)   # (C, C)
    scale = torch.sqrt(torch.tensor(C / (C - 1.0)))
    P_etf = scale * (H_ctr @ U_base)                      # (C, D)

    # Bước 3: Chuẩn hóa từng hàng về unit norm
    return F.normalize(P_etf, p=2, dim=1).cpu()


def procrustes_alignment(M: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """
    Tìm ma trận xoay R (D, D) sao cho M @ R gần P nhất theo Frobenius.
    Dùng svd_lowrank với q=C (rank thực sự của M.T@P là C) → chính xác & nhanh.
    """
    P  = P.to(M.device)
    C  = M.shape[0]           # số class hiện tại
    C_mat = M.T @ P           # (D, D) nhưng rank = C << D

    # svd_lowrank với q=C+4 và niter=4 cho kết quả cực kỳ chính xác
    # vì rank thực của C_mat đúng bằng C. Nhưng nếu D đủ nhỏ, linalg.svd sẽ tốt hơn tuyệt đối.
    if C_mat.shape[0] <= 1000:
        U, S, Vh = torch.linalg.svd(C_mat, full_matrices=False)
        V = Vh.T
    else:
        q = min(C + 4, C_mat.shape[0], C_mat.shape[1])
        U, S, V = torch.svd_lowrank(C_mat, q=q, niter=4)
        
    R = U @ V.T
    return R

# =============================================================================
# [YC2] Prototype Classifier — NCM + Cosine Similarity
#        Thay thế toàn bộ Ridge Regression + UnifiedCosineClassifier
# =============================================================================

def select_ridge_parameter(H, Y, lambdas=None):
    if lambdas is None:
        lambdas = [10**i for i in range(-5, 6)]
    N, D = H.shape
    best_lambda = lambdas[0]
    best_score = float('inf')
    
    # Sử dụng SVD để tính GCV (Nhanh và cực kỳ ổn định, chuẩn Fly-CL)
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    Z = U.T @ Y
    S2 = S ** 2
    
    Y_norm_sq = (Y ** 2).sum().item()
    Z_norm_sq = (Z ** 2).sum().item()
    base_error = Y_norm_sq - Z_norm_sq
    
    for l in lambdas:
        filter_factors = l / (S2 + l) # (K,) với K = min(N, D)
        Z_diff = filter_factors.unsqueeze(1) * Z
        error = base_error + (Z_diff ** 2).sum().item()
        
        tr_H_hat = len(S) - filter_factors.sum().item()
        denominator = (1 - tr_H_hat / N) ** 2
        gcv_score = error / (N * denominator)
        
        if gcv_score < best_score:
            best_score = gcv_score
            best_lambda = l
            
    return best_lambda

class UnifiedClassifier:
    def __init__(self, device: torch.device, args):
        self.device = device
        self.args = args
        self.classifier_type = args.classifier_type
        self.use_procrustes = args.use_procrustes
        self.num_prototypes = args.num_prototypes
        self.use_subspace = args.use_subspace
        self.use_acp = getattr(args, 'use_acp', False)
        self.subspace_rank = getattr(args, 'subspace_rank', 50)
        self.acp_alpha = getattr(args, 'acp_alpha', 1.0)
        
        if self.classifier_type == 'ridge_subspace' and not self.use_subspace:
            print("Warning: ridge_subspace selected but --use_subspace is False. Falling back to ridge.")
            self.classifier_type = 'ridge'
            
        self.prototypes = {}
        self.etf_prototypes = {}
        self.R = None
        self.V_r = None
        self.P_acp = None
        self._current_dim = 0
        
        # Cho Ridge
        self.G_global = None
        self.Q_global = None
        self.Wo = None
        self.class_counts = {}
        self.last_H = None
        self.last_Y = None
        self.last_lbl = None
        # Tích lũy toàn bộ H, Y cho GCV (tránh chọn lambda chỉ theo task hiện tại)
        self.H_accum = None
        self.Y_accum = None

    @property
    def current_dim(self) -> int:
        return self._current_dim

    def grow(self, num_new_neurons: int) -> None:
        pad = torch.zeros(self.num_prototypes, num_new_neurons, device=self.device)
        for cls_idx in self.prototypes:
            self.prototypes[cls_idx] = torch.cat(
                [self.prototypes[cls_idx], pad], dim=1
            )
        self._current_dim += num_new_neurons

    def update(self, H: torch.Tensor, labels: torch.Tensor, actual_classes: list[int]) -> None:
        H_dev = H.to(self.device)
        lbl = labels.to(self.device)
        
        self.last_H = H_dev
        self.last_lbl = lbl.clone()
        self.last_Y = torch.eye(self.args.num_classes, device=self.device)[lbl].float()
        
        # Cập nhật số lượng mẫu cho mỗi class
        for c in actual_classes:
            self.class_counts[c] = self.class_counts.get(c, 0) + (lbl == c).sum().item()
            
        # 1. Update prototypes (K-means cho NCM, hoặc mean bình thường)
        for cls in actual_classes:
            mask = (lbl == cls)
            if mask.sum() == 0:
                continue
            
            features_c = H_dev[mask]
            N_c = features_c.shape[0]
            
            if self.classifier_type == 'ncm':
                K_c = min(self.num_prototypes, N_c)
                if K_c <= 1:
                    protos = features_c.mean(dim=0, keepdim=True)
                else:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=K_c, n_init=10, random_state=42)
                    kmeans.fit(features_c.cpu().numpy())
                    protos = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=features_c.dtype)
                
                if K_c < self.num_prototypes:
                    mean_proto = features_c.mean(dim=0, keepdim=True)
                    pad = mean_proto.repeat(self.num_prototypes - K_c, 1)
                    protos = torch.cat([protos, pad], dim=0)
            else:
                # Với Ridge, chỉ cần 1 mean (để làm SVD)
                protos = features_c.mean(dim=0, keepdim=True)
                
            self.prototypes[cls] = protos.detach()
            
        if self._current_dim == 0:
            self._current_dim = H.shape[1]

        # 2. Update G_global và Q_global cho Ridge
        if self.classifier_type in ['ridge', 'ridge_subspace']:
            new_G = H_dev.T @ H_dev
            new_Q = H_dev.T @ self.last_Y
            
            if self.G_global is None:
                self.G_global = new_G
                self.Q_global = new_Q
            else:
                self.G_global += new_G
                self.Q_global += new_Q

            if self.H_accum is None:
                self.H_accum = H_dev
                self.Y_accum = self.last_Y
            else:
                self.H_accum = torch.cat([self.H_accum, H_dev], dim=0)
                self.Y_accum = torch.cat([self.Y_accum, self.last_Y], dim=0)

    def _global_class_means(self, sorted_cls: list[int]) -> torch.Tensor:
        """Global mean từ Q_global — nhất quán với G_global thay vì prototype cũ."""
        means = []
        for c in sorted_cls:
            n_c = self.class_counts[c]
            means.append(self.Q_global[:, c] / n_c)
        return torch.stack(means, dim=0).to(self.device)

    def _gcv_H_Y(self, P_proj: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """H, Y dùng cho GCV: toàn bộ dữ liệu đã thấy (có thể chiếu subspace)."""
        H_gcv = self.H_accum if self.H_accum is not None else self.last_H
        Y_gcv = self.Y_accum if self.Y_accum is not None else self.last_Y
        if P_proj is not None:
            H_gcv = H_gcv @ P_proj
        return H_gcv, Y_gcv

    def _argmax_seen_classes(self, logits: torch.Tensor) -> torch.Tensor:
        """Chỉ argmax trên class đã học — tránh chọn class chưa thấy (logit ≈ 0)."""
        sorted_cls = sorted(self.prototypes.keys())
        cls_t = torch.tensor(sorted_cls, dtype=torch.long, device=self.device)
        logits_seen = logits[:, cls_t]
        return cls_t[logits_seen.argmax(dim=1)].cpu()

    def align_prototypes(self) -> None:
        sorted_cls = sorted(self.prototypes.keys())
        num_classes = len(sorted_cls)

        if self.classifier_type == 'ncm':
            if not self.use_procrustes or num_classes < 3 or self._current_dim < num_classes - 1:
                return
            P = create_etf_prototypes(num_classes, self._current_dim).to(self.device)
            M_raw = torch.stack([self.prototypes[c].mean(dim=0) for c in sorted_cls], dim=0)
            M = F.normalize(M_raw.float(), p=2, dim=1).to(self.device)
            self.R = procrustes_alignment(M, P)
            self.etf_prototypes = {c: P[i].detach() for i, c in enumerate(sorted_cls)}
            
        elif self.classifier_type == 'ridge_subspace':
            # Subspace Extraction
            if num_classes < 2:
                # Quá ít lớp, fallback về ridge gốc hoặc identity
                self.V_r = None
                self.R = None
                self.P_acp = None
                H_gcv, Y_gcv = self._gcv_H_Y()
                best_lam = select_ridge_parameter(H_gcv, Y_gcv)
                print(f"    [Ridge Fallback] C={num_classes:3d}, best_lam={best_lam}")
                G_reg = self.G_global + best_lam * torch.eye(self.G_global.shape[0], device=self.device)
                L = torch.linalg.cholesky(G_reg)
                self.Wo = torch.cholesky_solve(self.Q_global, L)
                return
                
            # Global mean từ Q_global (khớp G_global, không dùng prototype cũ)
            M_raw = self._global_class_means(sorted_cls)  # (C, D)
            
            # SVD: U, S, V = torch.svd(M)
            U, S, Vh = torch.linalg.svd(M_raw, full_matrices=False)
            max_r = self.subspace_rank
            r = min(num_classes - 1, (S > 1e-6).sum().item(), max_r)
            if r <= 0: r = 1
            
            self.V_r = Vh[:r, :].T # (D, r)
            
            if self.use_procrustes and num_classes >= 3:
                P = create_etf_prototypes(num_classes, r).to(self.device)
                M_proj = M_raw @ self.V_r
                M_proj_norm = F.normalize(M_proj, p=2, dim=1)
                self.R = procrustes_alignment(M_proj_norm, P)
                self.P_acp = None
            elif self.use_acp and num_classes >= 2:
                self.R = None
                # Analytic Contrastive Projection (ACP)
                
                # Tính M_proj cho tất cả các lớp đã thấy (C x r)
                M_proj = M_raw @ self.V_r
                
                # Khôi phục chính xác Global Shared Covariance Sigma (r x r) từ G_global
                N_c = torch.tensor([self.class_counts[c] for c in sorted_cls], device=self.device).float()
                N_total = N_c.sum().item()
                
                # G_proj = V_r.T @ G_global @ V_r  (r x r)
                G_proj_exact = self.V_r.T @ self.G_global @ self.V_r
                
                # Between-class scatter: sum(N_c * mu_c * mu_c.T) = M_proj.T @ diag(N_c) @ M_proj
                S_b = M_proj.T @ (N_c.unsqueeze(1) * M_proj)
                
                # Within-class scatter = G_proj - S_b
                S_w = G_proj_exact - S_b
                Sigma = S_w / N_total
                
                # Đảm bảo tính khả nghịch
                Sigma = Sigma + 1e-6 * torch.eye(r, device=self.device)
                
                Us, Ss, _ = torch.linalg.svd(Sigma)
                Sigma_inv_half = Us @ torch.diag((Ss + 1e-6)**(-0.5)) @ Us.T
                Sigma_half = Us @ torch.diag(Ss**0.5) @ Us.T
                
                M_whitened = M_proj @ Sigma_inv_half # (C, r)
                
                Um, Sm, Vhm = torch.linalg.svd(M_whitened, full_matrices=False)
                # Tăng cường separation
                alpha = self.acp_alpha
                S_enhanced = Sm + alpha
                
                # P_target: (C, r)
                P_target = Um @ torch.diag(S_enhanced) @ Vhm @ Sigma_half
                
                # Học phép chiếu P (r x r) từ Global Data!
                # P_acp = (H_proj_all.T H_proj_all + lam_P I)^-1 H_proj_all.T Y_target_all
                # Với H_proj_all.T Y_target_all = (V_r.T Q_global) @ P_target
                
                # Pad P_target lên num_classes (100) để nhân được với Q_proj_exact (r x 100)
                P_target_padded = torch.zeros(self.args.num_classes, r, device=self.device)
                cls_indices = torch.tensor(sorted_cls, dtype=torch.long, device=self.device)
                P_target_padded[cls_indices] = P_target
                
                Q_proj_exact = self.V_r.T @ self.Q_global # (r, 100)
                Q_P = Q_proj_exact @ P_target_padded      # (r, r)
                
                lam_P = 0.1
                print(f"    [ACP Subspace] C={num_classes:3d}, r={r}, lam_P={lam_P}, alpha={alpha}")
                G_P = G_proj_exact + lam_P * torch.eye(r, device=self.device)
                L_P = torch.linalg.cholesky(G_P)
                self.P_acp = torch.cholesky_solve(Q_P, L_P)
                
            else:
                self.R = None
                self.P_acp = None
                
            # Tính P_proj
            if self.R is not None:
                P_proj = self.V_r @ self.R
            elif self.P_acp is not None:
                P_proj = self.V_r @ self.P_acp
            else:
                P_proj = self.V_r

                
            # Vi V_r và R thay đổi mỗi task, Q và G phải được cộng dồn ở không gian gốc 10000D, 
            # sau đó chiếu xuống không gian con mới bằng P_proj để đảm bảo tính đúng đắn toán học.
            G_proj = P_proj.T @ self.G_global @ P_proj # (r, r)
            Q_proj = P_proj.T @ self.Q_global # (r, C)
            
            # GCV trên toàn bộ dữ liệu đã thấy trong subspace hiện tại
            H_gcv, Y_gcv = self._gcv_H_Y(P_proj)
            best_lam = select_ridge_parameter(H_gcv, Y_gcv)
            
            print(f"    [Ridge Subspace] C={num_classes:3d}, r={r}, best_lam={best_lam}")
            
            G_reg = G_proj + best_lam * torch.eye(G_proj.shape[0], device=self.device)
            L = torch.linalg.cholesky(G_reg)
            self.Wo = torch.cholesky_solve(Q_proj, L)
            
        elif self.classifier_type == 'ridge':
            # Giải Ridge gốc 10000D
            H_gcv, Y_gcv = self._gcv_H_Y()
            best_lam = select_ridge_parameter(H_gcv, Y_gcv)
            print(f"    [Ridge] C={num_classes:3d}, Full D={self._current_dim}, best_lam={best_lam}")
            G_reg = self.G_global + best_lam * torch.eye(self.G_global.shape[0], device=self.device)
            try:
                L = torch.linalg.cholesky(G_reg)
                self.Wo = torch.cholesky_solve(self.Q_global, L)
            except torch.linalg.LinAlgError:
                self.Wo = torch.linalg.solve(G_reg, self.Q_global)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2 and z.shape[1] != self._current_dim:
            z = z.T

        z_float = z.float().to(self.device)

        if self.classifier_type == 'ncm':
            z_norm = F.normalize(z_float, p=2, dim=1)
            sorted_cls = sorted(self.prototypes.keys())
            
            if self.use_procrustes and self.R is not None and len(self.etf_prototypes) > 0:
                P_etf = torch.stack([self.etf_prototypes[c] for c in sorted_cls], dim=0)
                P = P_etf @ self.R.T
                P_norm = F.normalize(P, p=2, dim=1)
                sim = z_norm @ P_norm.T
                local_preds = sim.argmax(dim=1)
            else:
                P = torch.cat([self.prototypes[c] for c in sorted_cls], dim=0).float()
                P_norm = F.normalize(P, p=2, dim=1)
                sim = z_norm @ P_norm.T
                sim = sim.view(z_norm.shape[0], len(sorted_cls), self.num_prototypes)
                sim_max, _ = sim.max(dim=2)
                local_preds = sim_max.argmax(dim=1)
                
            cls_t = torch.tensor(sorted_cls, dtype=torch.long, device=self.device)
            return cls_t[local_preds].cpu()
            
        elif self.classifier_type == 'ridge_subspace':
            if self.V_r is not None:
                H_test_proj = z_float @ self.V_r
                if self.R is not None:
                    H_test_aligned = H_test_proj @ self.R
                elif self.P_acp is not None:
                    H_test_aligned = H_test_proj @ self.P_acp
                else:
                    H_test_aligned = H_test_proj
                logits = H_test_aligned @ self.Wo
            else:
                logits = z_float @ self.Wo
            return self._argmax_seen_classes(logits)
            
        elif self.classifier_type == 'ridge':
            logits = z_float @ self.Wo
            return self._argmax_seen_classes(logits)

    def evaluate(self, H: torch.Tensor, labels: torch.Tensor) -> float:
        if H.is_sparse:
            H = H.to_dense()
        preds = self.predict(H)
        return (preds == labels.cpu()).float().mean().item() * 100.0


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    random_initialization(args.seed)

    # [YC3] data_augmentation='vit' → load_dataset.py sẽ dùng
    #        Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) → ảnh về [-1,1]
    print(f"[YC3] data_augmentation='{args.data_augmentation}' → "
          f"{'[-1,1] normalization (Paper Appendix C.3)' if args.data_augmentation == 'vit' else 'other'}")

    if args.dataset in ("CIFAR-100", "CUB-200-2011", "VTAB"):
        print("Load and Split CIL Dataset...")
        train_loader, test_loader = load_dataset(args)
        print("Load and Split CIL Dataset Done")

    pretrained_model = load_model(args.model_name)
    pretrained_model.out_dim = args.embedding_dim
    pretrained_model.eval()
    pretrained_model.to(device)

    # --- Khởi tạo tĩnh (Thí nghiệm 1: Tắt Growable) ---
    non_zero_per_col  = args.synaptic_degree
    
    print(f"Khởi tạo tĩnh ma trận chiếu D_total = {args.expand_dim}...")
    projection_matrix = _make_sparse_projection_block(
        num_neurons=args.expand_dim,
        embedding_dim=args.embedding_dim,
        non_zero_per_row=non_zero_per_col,
        device=device
    )
    
    # Coi toàn bộ ma trận như 1 block duy nhất
    block_sizes = [args.expand_dim]

    # [YC2] Prototype Classifier — thay thế Ridge + UnifiedCosineClassifier
    classifier = UnifiedClassifier(device=device, args=args)
    classifier._current_dim = args.expand_dim  # Set cứng không gian 10000 chiều ngay từ đầu

    acc                  = {}
    training_time        = []
    feature_extract_time = []

    print(f"\nConfig: D_total={args.expand_dim} (static) | "
          f"coding_level={args.coding_level}/block | "
          f"synaptic_degree={args.synaptic_degree}")
    print("Start Continual Learning\n")

    for task in range(args.num_tasks):
        acc[task]      = []
        training_start = time.time()

        # --- Feature extraction (backbone frozen) ---
        t0 = time.time()
        train_embeddings, train_labels = feature_extract(
            pretrained_model, train_loader[task], device
        )
        feature_extract_time.append(time.time() - t0)

        # --- Bỏ qua Fisher-adaptive (Thí nghiệm 1) ---
        print(f"[Task {task:02d}] Sử dụng ma trận chiếu tĩnh (D_total={args.expand_dim})")
        # D_total luôn bằng expand_dim, không grow nữa

        # --- Projection + [YC1] Block-wise Absolute WTA (Train) ---
        proj_sparse  = projection_matrix.to_sparse_csc()
        H_full_train = torch.sparse.mm(proj_sparse, train_embeddings.T)  # (D, N)

        # [YC1] WTA per-block với absolute top-k (giữ nguyên dấu âm/dương)
        H_wta_train  = blockwise_wta(H_full_train, block_sizes, args.coding_level)
        del H_full_train

        # --- [YC2] Cập nhật Prototype cho các class mới của task này ---
        actual_classes = sorted([int(c) for c in train_labels.unique().cpu().tolist()])
        classifier.update(H_wta_train.T.contiguous(), train_labels, actual_classes)
        classifier.align_prototypes()
        del H_wta_train, train_embeddings

        torch.cuda.empty_cache()
        training_time.append(time.time() - training_start)

        # --- Evaluation trên tất cả sub-tasks đã học ---
        for sub_task in range(task + 1):
            test_embeddings, test_labels = feature_extract(
                pretrained_model, test_loader[sub_task], device
            )
            H_full_test = torch.sparse.mm(proj_sparse, test_embeddings.T)  # (D, N_test)
            del test_embeddings

            # [YC1] Cùng block-wise absolute WTA — block_sizes giống train
            H_wta_test  = blockwise_wta(H_full_test, block_sizes, args.coding_level)
            del H_full_test

            # [YC2] Predict bằng Cosine Similarity với Prototypes
            acc_val = classifier.evaluate(H_wta_test.T, test_labels)
            del H_wta_test
            acc[sub_task].append(acc_val)

        del proj_sparse
        torch.cuda.empty_cache()
        print(f"[Task {task:02d}] Done | train_time={training_time[-1]:.2f}s\n")

    # =================================================================
    # Hiển thị kết quả
    # =================================================================
    acc_matrix = [[0.0] * args.num_tasks for _ in range(args.num_tasks)]
    for i, (task_i, vals) in enumerate(acc.items()):
        for j, v in enumerate(vals):
            acc_matrix[i][i + j] = round(v, 2)

    print("\n" + "=" * 60)
    print("\nAccuracy Matrix")
    for i in range(args.num_tasks):
        row = []
        for j in range(args.num_tasks):
            if i > j:
                row.append('0.00')
            else:
                row.append(acc_matrix[i][j])
        print(row)

    print("\nAverage Accuracy")
    A_t = []
    for j in range(args.num_tasks):
        cnt = sum(acc_matrix[i][j] for i in range(j + 1)) / (j + 1)
        A_t.append(round(cnt, 2))
    print(", ".join(str(x) for x in A_t) + ", ")

    print("\nAccumulated Accuracy")
    print(f"{round(float(np.mean(A_t)), 2)}")

    print("\nTraining Time")
    print(", ".join(str(round(t, 2)) for t in training_time) + ", ")

    print("\nAverage Training Time")
    print(f"{round(float(np.mean(training_time)), 2)}")

    print("\nFeature Extract Time")
    print(", ".join(str(round(t, 2)) for t in feature_extract_time) + ", ")

    print("\nAverage Feature Extract Time")
    print(f"{round(float(np.mean(feature_extract_time)), 2)}")