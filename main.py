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

import torch
import numpy as np
from torch.nn import functional as F

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

    # [ETF] Bật/Tắt căn chỉnh Procrustes
    parser.add_argument('--use_procrustes', action='store_true',
                        help='Sử dụng ETF Prototypes và Procrustes Alignment')

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
    """Tạo ma trận ETF (C, D) ngẫu nhiên -> chuẩn hóa -> SVD trực giao hóa."""
    # Khi C=1, trả về 1 vector bất kỳ có norm=1
    if num_classes == 1:
        return F.normalize(torch.randn(1, feat_dim), p=2, dim=1).cpu()

    random_matrix = torch.randn(num_classes, feat_dim)
    # Trừ đi mean theo cột để có tổng bằng 0
    centered_matrix = random_matrix - random_matrix.mean(dim=0, keepdim=True)
    # SVD
    U, S, Vh = torch.linalg.svd(centered_matrix, full_matrices=False)
    # Trực giao hóa
    orthogonal_matrix = U @ Vh
    # Chuẩn hóa L2 norm cho từng hàng (prototype)
    etf_prototypes = F.normalize(orthogonal_matrix, p=2, dim=1)
    return etf_prototypes.cpu()

def procrustes_alignment(M: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Tìm ma trận xoay R (D, D) sao cho M @ R gần P nhất."""
    P = P.to(M.device)
    C_mat = M.T @ P
    U, _, Vh = torch.linalg.svd(C_mat, full_matrices=False)
    R = U @ Vh
    return R

# =============================================================================
# [YC2] Prototype Classifier — NCM + Cosine Similarity
#        Thay thế toàn bộ Ridge Regression + UnifiedCosineClassifier
# =============================================================================
class PrototypeClassifier:
    """
    [YC2] Nearest-Class-Mean Classifier với Cosine Similarity.

    Lưu prototype μ_c = mean(h | y=c) cho mỗi class c đã thấy.
    Khi thêm block mới (D tăng), zero-pad prototype cũ để nhất quán chiều.
    Predict: ŷ = argmax_c cosine(h, μ_c).

    Ưu điểm so với Ridge trong kiến trúc grow:
      • Không có direction bias khi zero-pad (cosine normalize 2 vế).
      • Không cần hyperparameter λ — không cần GCV.
      • Training O(N·D), không O(N²·D).
      • Block-wise WTA đã làm các block trực giao → NCM đủ mạnh.
    """

    def __init__(self, device: torch.device, use_procrustes: bool = False):
        self.device = device
        self.use_procrustes = use_procrustes
        # global_class_idx → prototype tensor (D,)
        self.prototypes: dict[int, torch.Tensor] = {}
        self.etf_prototypes: dict[int, torch.Tensor] = {}
        self.R: torch.Tensor | None = None
        self._current_dim: int = 0

    @property
    def current_dim(self) -> int:
        return self._current_dim

    def grow(self, num_new_neurons: int) -> None:
        """
        [YC2 - Zero-Padding] Khi projection mở rộng thêm num_new_neurons chiều,
        zero-pad cuối mỗi prototype cũ để giữ nhất quán số chiều D.
        """
        pad = torch.zeros(num_new_neurons, device=self.device)
        for cls_idx in self.prototypes:
            self.prototypes[cls_idx] = torch.cat(
                [self.prototypes[cls_idx], pad], dim=0
            )
        self._current_dim += num_new_neurons

    def update(
        self,
        H: torch.Tensor,            # (N, D) dense — sau block-wise WTA
        labels: torch.Tensor,       # (N,) global class labels
        actual_classes: list[int],  # global class idx của task hiện tại
    ) -> None:
        """
        Tính μ_c = mean(H[y==c]) và lưu vào prototypes[c].
        Mỗi class chỉ xuất hiện trong 1 task → không cần cộng dồn.
        """
        H_dev = H.to(self.device)
        lbl   = labels.to(self.device)
        for cls in actual_classes:
            mask = (lbl == cls)
            if mask.sum() == 0:
                continue
            self.prototypes[cls] = H_dev[mask].mean(dim=0).detach()
        if self._current_dim == 0:
            self._current_dim = H.shape[1]

    def align_prototypes(self) -> None:
        """
        Khởi tạo ETF Prototypes và tìm ma trận xoay R để gióng hàng
        Empirical Means (M) khớp với ETF Prototypes (P).
        Chỉ chạy nếu tính năng này được bật (use_procrustes = True).
        """
        if not self.use_procrustes:
            return

        sorted_cls = sorted(self.prototypes.keys())
        num_classes = len(sorted_cls)
        
        # Nếu số lớp < 3, Procrustes có thể không ổn định, bỏ qua
        if num_classes < 3 or self._current_dim < num_classes - 1:
            return
            
        # Tạo ETF Prototypes P (C, D)
        P = create_etf_prototypes(num_classes, self._current_dim).to(self.device)
        
        # Tạo Empirical Means M (C, D)
        M = torch.stack([self.prototypes[c] for c in sorted_cls], dim=0).to(self.device)
        
        # Tìm ma trận xoay R bằng Procrustes
        self.R = procrustes_alignment(M, P)
        
        # Lưu lại ETF Prototypes tương ứng cho từng class
        self.etf_prototypes = {c: P[i].detach() for i, c in enumerate(sorted_cls)}

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        [YC2] ŷ = argmax_c cosine(h, μ_c).

        Args:
            z : (N, D) hoặc (D, N).
        Returns:
            global class indices (N,) on CPU.
        """
        if z.dim() == 2 and z.shape[1] != self._current_dim:
            z = z.T   # (D, N) → (N, D)

        z_float = z.float().to(self.device)

        if self.R is not None and len(self.etf_prototypes) > 0:
            # Dùng ETF Prototypes và áp dụng ma trận xoay R
            sorted_cls = sorted(self.etf_prototypes.keys())
            P = torch.stack([self.etf_prototypes[c] for c in sorted_cls], dim=0)
            # Biến đổi z: z_aligned = z @ R
            z_float = z_float @ self.R
        else:
            # Dùng Empirical Means tạm thời
            sorted_cls = sorted(self.prototypes.keys())
            P = torch.stack([self.prototypes[c] for c in sorted_cls], dim=0).float()

        z_norm = F.normalize(z_float, p=2, dim=1)   # (N, D)
        P_norm = F.normalize(P,       p=2, dim=1)   # (C, D)
        sim    = z_norm @ P_norm.T                  # (N, C)

        local_preds = sim.argmax(dim=1)
        cls_t       = torch.tensor(sorted_cls, dtype=torch.long, device=self.device)
        return cls_t[local_preds].cpu()

    def evaluate(self, H: torch.Tensor, labels: torch.Tensor) -> float:
        """Wrapper: trả về accuracy (%)."""
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
    classifier = PrototypeClassifier(device=device, use_procrustes=args.use_procrustes)
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