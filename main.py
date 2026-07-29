import argparse
import time

import torch
import timm
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from datasets.load_dataset import load_dataset
from models.load_model import load_model
from utils import random_initialization, feature_extract, target2onehot


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Input hyperparameters for the experiment.")

    # Continual Learning Task Setting
    parser.add_argument('--dataset', default='CIFAR-100', help='Choose dataset')
    parser.add_argument('--root', default='../data', help='Dataset path')
    parser.add_argument('--num_classes', type=int, default=100, help='Total number of classes')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks')

    # model Architecture
    parser.add_argument('--model_name', type=str, default="vit_base_patch16_224", help='model name')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension of pre-trained model')
    parser.add_argument('--expand_dim', type=int, default=10000, help='Expansion dimension of FlyModel')
    parser.add_argument('--synaptic_degree', type=int, default=100, help='Number of connections')
    parser.add_argument('--coding_level', type=float, default=0.01, help='Top-k number')

    # Training Configuration
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--ridge_lower', type=float, default=4, help='lower bound for ridge coefficient (log10)')
    parser.add_argument('--ridge_upper', type=float, default=10, help='lower bound for ridge coefficient (log10)')
    parser.add_argument('--data_augmentation', default=None, help='choose which normalization or not')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='Choose gpu')
    
    return parser


def select_ridge_parameter(Features, Y, ridge_lower, ridge_upper):
    X = Features
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_sq = S**2
    UTY = U.T @ Y
    ridges = torch.tensor(10.0 ** np.arange(ridge_lower, ridge_upper), device=X.device)
    n_samples = X.shape[0]
    
    gcv_scores = []
    for ridge in ridges:
        diag = S_sq / (S_sq + ridge)
        df = diag.sum()
        Y_hat = U @ (diag[:, None] * UTY)
        residual = torch.norm(Y - Y_hat)**2
        gcv = (residual / n_samples) / (1 - df / n_samples)**2
        gcv_scores.append(gcv.item())

    optimal_idx = np.argmin(gcv_scores)
    return ridges[optimal_idx]


def _make_sparse_projection_block(
    num_neurons: int,
    embedding_dim: int,
    non_zero_per_row: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Tạo một block ma trận chiếu thưa (sparse random projection block).

    Mỗi hàng (neuron) kết nối ngẫu nhiên với `non_zero_per_row` chiều đầu vào.
    Block này sẽ được nối hàng (row-wise) vào ma trận chiếu hiện tại khi có task mới.

    Args:
        num_neurons     : Số neuron mới (số hàng của block, tức m).
        embedding_dim   : Số chiều đầu vào d_in (số cột, cố định).
        non_zero_per_row: Số kết nối không-zero mỗi hàng (synaptic_degree).
        device          : torch device.

    Returns:
        block : Tensor dense shape (num_neurons, embedding_dim) — trả về dense,
                gọi .to_sparse_csc() bên ngoài khi cần.
    """
    block = torch.zeros(num_neurons, embedding_dim)
    for row in range(num_neurons):
        selected_cols = torch.randperm(embedding_dim)[:non_zero_per_row]
        block[row, selected_cols] = torch.randn(non_zero_per_row)
    return block.to(device)


# =============================================================================
# TODO: RAPID - [Single Unified Cosine Classifier]
# Thay thế hệ thống per-task head bằng một classifier toàn cục duy nhất.
# Thiết kế:
#   - W: ma trận trọng số (expand_dim x num_seen_classes), được grow khi có class mới.
#   - class_centroids: danh sách tensor prototype cho mỗi class đã học.
#   - fit(): cập nhật cột của W bằng nghiệm Ridge cho các class của task hiện tại.
#   - expand(): nối thêm các cột zero vào W để chứa class mới.
#   - predict(): L2-normalize z và W rồi tính cosine similarity → argmax.
# =============================================================================
class UnifiedCosineClassifier:
    """
    Single Unified Cosine Classifier cho Continual Learning.

    Không yêu cầu task_id khi inference — predict() trả về global class index
    trực tiếp dựa trên cosine similarity giữa feature vector và tất cả class
    prototypes đã học.
    """

    def __init__(self, init_dim: int, num_total_classes: int, device: torch.device):
        """
        Args:
            init_dim         : Số chiều ban đầu của không gian fly-projection (k).
                               Sẽ tăng dần sau mỗi task qua grow_projection().
            num_total_classes: Tổng số class tối đa (dùng để pre-allocate cột W).
            device           : torch device.
        """
        self.num_total_classes = num_total_classes
        self.device = device
        # fitted_class_indices: danh sách global class idx đã được fit (thứ tự thêm vào)
        self.fitted_class_indices: list[int] = []

        self._current_dim: int = init_dim

        # W[:, global_class_idx] chứa trọng số của class đó sau khi fit()
        self.W = torch.zeros(init_dim, num_total_classes, device=device)  # (k, C_total)

    # ------------------------------------------------------------------
    # Property: chiều hiện tại của feature space
    # ------------------------------------------------------------------
    @property
    def current_dim(self) -> int:
        """Số chiều hiện tại của không gian chiếu (số hàng của W)."""
        return self._current_dim

    # ------------------------------------------------------------------
    # grow_projection: mở rộng W theo chiều hàng + zero-pad class cũ
    # ------------------------------------------------------------------
    def grow_projection(self, num_new_neurons: int) -> None:
        """
        Mở rộng ma trận W khi không gian chiếu được mở rộng thêm m neuron mới.

        Logic zero-padding:
          - Các cột class cũ ([:num_seen_classes]): pad thêm `num_new_neurons` hàng 0
            vì các class cũ không được huấn luyện trên chiều mới → trọng số = 0.
          - Các cột class mới (sẽ được điền bởi fit()): tự nhiên có toàn bộ
            (current_dim + num_new_neurons) hàng ngay từ đầu.

        Args:
            num_new_neurons : Số neuron mới thêm vào (m), kích thước block mới.
        """
        # Zero-pad W cũ: nối thêm num_new_neurons hàng 0 cho tất cả cột
        # Shape: (current_dim, C_total) → (current_dim + m, C_total)
        padding = torch.zeros(num_new_neurons, self.num_total_classes, device=self.device)
        self.W = torch.cat([self.W, padding], dim=0)  # (D_new, C_total)
        self._current_dim += num_new_neurons
        # Kết quả:
        # - Cột class cũ (đã được fit): hàng [D_old:D_new] = 0  ← zero-padded
        # - Cột class mới (chưa fit):   toàn 0, sẽ được fit() ghi vào sau

    # ------------------------------------------------------------------
    # expand: no-op, kept for API compatibility
    # ------------------------------------------------------------------
    def expand(self, num_new_classes: int) -> None:
        """No-op: class tracking done automatically inside fit()."""
        pass

    # ------------------------------------------------------------------
    # Fit: ghi nghiệm Ridge vào đúng cột W theo global class index
    # ------------------------------------------------------------------
    def fit(
        self,
        Wo_task: torch.Tensor,          # (D, K) — nghiệm Ridge
        task_class_indices: list[int],  # global class indices (KHÔNG cần sequential)
        train_embeddings=None,          # unused, kept for compat
        train_labels=None,              # unused, kept for compat
    ) -> None:
        """
        Ghi W[:, global_idx] = Wo_task[:, local_idx] cho từng class.
        Không giả định classes là sequential — dùng đúng global index.
        """
        for local_idx, global_idx in enumerate(task_class_indices):
            self.W[:, global_idx] = Wo_task[:, local_idx]
        for g in task_class_indices:
            if g not in self.fitted_class_indices:
                self.fitted_class_indices.append(g)

    # ------------------------------------------------------------------
    # Predict: Cosine Classifier — không cần task_id, trả về GLOBAL class idx
    # ------------------------------------------------------------------
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán global class label bằng Cosine Similarity.
        Chỉ so sánh với các class đã được fit (fitted_class_indices).
        Trả về global class index (CPU tensor).
        """
        # Xử lý shape: (D, N) → (N, D)
        # Dùng .shape[1] vì không thể biết chiều nào là D khi D==N
        if z.dim() == 2 and z.shape[0] == self._current_dim:
            z = z.T  # (D, N) -> (N, D)

        z_norm = F.normalize(z.float(), p=2, dim=1)  # (N, D)

        # Lấy đúng cột của các class đã fit (KHÔNG dùng :num_seen_classes)
        fitted  = sorted(self.fitted_class_indices)
        W_active = self.W[:, fitted]                           # (D, num_fitted)
        W_norm   = F.normalize(W_active.float(), p=2, dim=0)  # (D, num_fitted)

        cosine_logits = z_norm @ W_norm                        # (N, num_fitted)
        local_preds   = cosine_logits.argmax(dim=1)            # (N,) — index vào fitted

        # Ánh xạ local index → global class index
        fitted_t = torch.tensor(fitted, dtype=torch.long, device=self.device)
        global_preds = fitted_t[local_preds].cpu()  # (N,) global idx, trên CPU
        return global_preds

    # ------------------------------------------------------------------
    # Evaluate: wrapper trả về accuracy (%)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_embeddings: torch.Tensor,  # (D, N)
        test_labels: torch.Tensor,      # (N,) global labels
    ) -> float:
        if test_embeddings.is_sparse:
            test_embeddings = test_embeddings.to_dense()
        predicts = self.predict(test_embeddings)  # (N,) global idx, CPU
        return (predicts == test_labels.cpu()).float().mean().item() * 100.0


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if cuda_available else "cpu")
    random_initialization(args.seed)

    if args.dataset == "CIFAR-100" or args.dataset == "CUB-200-2011" or args.dataset == "VTAB":
        print("Load and Split CIL Dataset...")
        train_loader, test_loader = load_dataset(args)
        print("Load and Split CIL Dataset Done")

    pretrained_model = load_model(args.model_name)
    pretrained_model.out_dim = args.embedding_dim
    pretrained_model.eval()
    pretrained_model.to(device)
    
    # TODO: RAPID - [Random Projection Matrix Init — Growable]
    # Task 1: khởi tạo block chiếu ban đầu kích thước (expand_dim × embedding_dim).
    # Task t>1: sinh block mới (expand_dim × embedding_dim) và nối row-wise vào P,
    #           tạo không gian (t * expand_dim) × embedding_dim.
    # Biến projection_matrix luôn là dense tensor; chuyển sang sparse khi dùng.
    non_zero_per_col = args.synaptic_degree
    projection_matrix = _make_sparse_projection_block(
        num_neurons=args.expand_dim,
        embedding_dim=args.embedding_dim,
        non_zero_per_row=non_zero_per_col,
        device=device,
    )  # Dense (expand_dim, embedding_dim) — block của Task 1

    acc = {}
    training_time = []
    feature_extract_time = []

    # Ridge Regression dùng per-task (không cần global Q/G vì classes không sequential)

    # TODO: RAPID - [Unified Classifier Init]
    # Khởi tạo Single Unified Cosine Classifier thay thế toàn bộ per-task head.
    # init_dim = expand_dim: số neuron ban đầu (k), W sẽ grow qua grow_projection().
    num_classes_per_task = args.num_classes // args.num_tasks
    classifier = UnifiedCosineClassifier(
        init_dim=args.expand_dim,
        num_total_classes=args.num_classes,
        device=device,
    )
    print("Start Continual Learning")
    for task in range(args.num_tasks):
        acc[task] = []
        training_start = time.time()

        # TODO: RAPID - [Projection Matrix Grow — Task t > 0]
        # Khi sang task mới, sinh thêm một block chiếu (expand_dim × embedding_dim)
        # và nối row-wise vào projection_matrix hiện tại:
        #   P_{t} = cat([P_{t-1}, new_block], dim=0)  → shape: (t*k × d_in)
        # Đồng thời, grow W trong classifier: zero-pad m hàng mới cho class cũ.
        if task > 0:
            new_block = _make_sparse_projection_block(
                num_neurons=args.expand_dim,
                embedding_dim=args.embedding_dim,
                non_zero_per_row=non_zero_per_col,
                device=device,
            )  # (expand_dim, embedding_dim)
            # Nối row-wise: projection_matrix: (t*k, d_in)
            projection_matrix = torch.cat([projection_matrix, new_block], dim=0)
            # Zero-pad W trong classifier cho m chiều mới (class cũ = 0 trên chiều mới)
            classifier.grow_projection(num_new_neurons=args.expand_dim)
            print(f"[Task {task}] Projection expanded: {projection_matrix.shape[0]} neurons "
                  f"| Classifier W: {classifier.current_dim} dims")

        # Kích thước hiện tại của không gian chiếu
        current_proj_dim = projection_matrix.shape[0]  # t * expand_dim
        # Số neuron WTA top-k (giữ coding_level% trên tổng chiều hiện tại)
        topk_neurons = max(1, int(current_proj_dim * args.coding_level))

        feature_extract_start = time.time()
        train_embeddings, train_labels = feature_extract(pretrained_model, train_loader[task], device)
        feature_extract_end = time.time()
        feature_extract_time.append(feature_extract_end - feature_extract_start)

        # TODO: RAPID - [Feature Extraction via Projection (Train)]
        # Bước 1: chiếu lên không gian (current_proj_dim × d_in) — đã grow.
        # Bước 2: WTA top-k trên toàn bộ current_proj_dim neuron.
        # Lưu ý: projection_matrix là dense tensor, chuyển sang sparse khi nhân.
        proj_sparse = projection_matrix.to_sparse_csc()  # (current_proj_dim, d_in)
        train_embeddings = torch.sparse.mm(proj_sparse, train_embeddings.T)  # (current_proj_dim, N)
        values, indices = train_embeddings.topk(topk_neurons, dim=0, largest=True)
        output = torch.zeros_like(train_embeddings)
        output.scatter_(0, indices, values)
        train_embeddings = output  # (current_proj_dim, N)

        # ----------------------------------------------------------------
        # Lấy ACTUAL global class indices từ data (load_dataset dùng random.sample
        # nên thứ tự class hoàn toàn ngẫu nhiên — KHÔNG giả định sequential)
        # ----------------------------------------------------------------
        actual_classes   = sorted([int(c) for c in train_labels.unique().cpu().tolist()])
        num_task_classes = len(actual_classes)

        # Map global label → local index [0, K-1] cho Ridge Regression
        g2l = {g: l for l, g in enumerate(actual_classes)}
        local_labels = torch.tensor(
            [g2l[int(lb)] for lb in train_labels.cpu().tolist()],
            dtype=torch.long, device=device
        )
        Y_local = target2onehot(local_labels, num_task_classes)  # (N, K) — safe!

        # Per-task Ridge Regression (closed-form, không cần global Q/G)
        D      = current_proj_dim
        G_task = train_embeddings @ train_embeddings.T  # (D, D)
        Q_task = train_embeddings @ Y_local             # (D, K)
        ridge  = select_ridge_parameter(train_embeddings.T, Y_local, args.ridge_lower, args.ridge_upper)
        L      = torch.linalg.cholesky(G_task + ridge * torch.eye(D, device=device))
        Wo_task = torch.cholesky_solve(Q_task, L)       # (D, K)
        training_end = time.time()
        training_time.append(training_end - training_start)

        # Ghi nghiệm Ridge vào đúng cột W (theo actual global class indices)
        classifier.expand(num_new_classes=num_task_classes)
        classifier.fit(Wo_task=Wo_task, task_class_indices=actual_classes)

        for sub_task in range(task + 1):
            test_embeddings, test_labels = feature_extract(pretrained_model, test_loader[sub_task], device)
            # TODO: RAPID - [Feature Extraction via Projection (Test/Eval)]
            # Dùng projection_matrix đã expanded (current_proj_dim × d_in).
            # WTA top-k nhất quán với lúc train: topk_neurons trên current_proj_dim.
            # Classifier.evaluate() nhận (current_proj_dim, N) — khớp với W.shape[0].
            test_embeddings = torch.sparse.mm(proj_sparse, test_embeddings.T)  # (current_proj_dim, N)
            values, indices = test_embeddings.topk(topk_neurons, dim=0, largest=True)
            output = torch.zeros_like(test_embeddings)
            output.scatter_(0, indices, values)
            test_embeddings_dense = output  # (current_proj_dim, N)

            # TODO: RAPID - [Unified Classifier: Evaluate — không cần task_id]
            # Cosine Classifier: z và W[:current_proj_dim, :seen_classes] đồng kích thước.
            test_accuracy = classifier.evaluate(test_embeddings_dense, test_labels)
            acc[sub_task].append(test_accuracy)

    # display acc_matrix
    acc_matrix = [["{:.2f}".format(0.00) for _ in range(args.num_tasks)] for _ in range(len(acc))]
    for i, (task, values) in enumerate(acc.items()):
        for j, value in enumerate(values):
            acc_matrix[i][i + j] = round(value, 2)
    
    print("Accuracy Matrix")
    for row in acc_matrix:
        print(row)
    print()

    print("Average Accuracy")
    A_t = []
    for j in range(args.num_tasks):
        cnt = 0.0
        for i in range(j + 1):
            cnt += acc_matrix[i][j]
        cnt /= (j + 1)
        A_t.append(cnt)
        print(round(cnt, 2), end=", ")
    print("\n")

    print("Accumulated Accuracy")
    print(round(np.mean(A_t), 2))
    print()

    print("Training Time")
    for task_time in training_time:
        print(round(task_time, 2), end=", ")
    print("\n")

    print("Average Training Time")
    print(round(np.mean(training_time), 2))
    print()

    print("Feature Extract Time")
    for task_time in feature_extract_time:
        print(round(task_time, 2), end=", ")
    print("\n")

    print("Average Feature Extract Time")
    print(round(np.mean(feature_extract_time), 2))
    print()