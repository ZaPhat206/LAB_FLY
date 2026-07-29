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
    ridges = torch.tensor(10.0 ** np.arange(ridge_lower, ridge_upper))
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
        self.num_seen_classes = 0          # Số class đã học đến hiện tại
        self.class_centroids: list[torch.Tensor] = []  # prototype mỗi class

        # current_dim: số hàng hiện tại của W = tổng số neuron đã khởi tạo.
        # Bắt đầu bằng k (init_dim), tăng m mỗi task qua grow_projection().
        self._current_dim: int = init_dim

        # W shape: (current_dim, num_total_classes) — sẽ được grow theo chiều hàng.
        # Chỉ cột [:num_seen_classes] có nghĩa tại bất kỳ thời điểm nào.
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
        # - Cột class cũ [:num_seen_classes]: hàng [D_old:D_new] = 0  ← zero-padded
        # - Cột class mới (chưa điền): toàn 0, sẽ được fit() ghi vào sau

    # ------------------------------------------------------------------
    # expand_classes: ghi nhận class mới (chỉ cập nhật con trỏ)
    # ------------------------------------------------------------------
    def expand(self, num_new_classes: int) -> None:
        """
        Đăng ký thêm `num_new_classes` class mới vào classifier.
        Các cột tương ứng trong W sẽ được điền bởi fit().
        """
        assert self.num_seen_classes + num_new_classes <= self.num_total_classes, (
            f"Vượt quá num_total_classes={self.num_total_classes}. "
            f"Đang cố thêm {num_new_classes} class, hiện có {self.num_seen_classes}."
        )
        self.num_seen_classes += num_new_classes

    # ------------------------------------------------------------------
    # Fit: ghi nghiệm Ridge vào đúng cột của W, lưu centroid
    # ------------------------------------------------------------------
    def fit(
        self,
        Wo_task: torch.Tensor,       # (expand_dim, num_classes_this_task) — nghiệm Ridge
        task_class_indices: list[int],  # global index của từng class trong task này
        train_embeddings: torch.Tensor, # (expand_dim, N) — projected features (sparse OK)
        train_labels: torch.Tensor,     # (N,) — global class labels
    ) -> None:
        """
        Ghi cột nghiệm Ridge Wo_task vào ma trận W toàn cục,
        đồng thời tính và lưu class centroid (prototype) cho mỗi class mới.

        Args:
            Wo_task           : (D, K) — nghiệm Ridge cho K class của task hiện tại.
            task_class_indices: list global class id, len = K.
            train_embeddings  : (D, N) — projected & WTA features, dùng để tính centroid.
            train_labels      : (N,)   — global label tương ứng với mỗi sample.
        """
        # Ghi cột nghiệm vào W
        for local_idx, global_idx in enumerate(task_class_indices):
            self.W[:, global_idx] = Wo_task[:, local_idx]

        # Tính và lưu centroid (mean feature) cho mỗi class
        # Dense embedding để tính mean dễ hơn
        if train_embeddings.is_sparse:
            dense_emb = train_embeddings.to_dense()   # (D, N)
        else:
            dense_emb = train_embeddings              # (D, N)

        for global_idx in task_class_indices:
            mask = (train_labels == global_idx)        # (N,)
            if mask.sum() == 0:
                centroid = torch.zeros(self._current_dim, device=self.device)
            else:
                centroid = dense_emb[:, mask].mean(dim=1)  # (D,)
                centroid = F.normalize(centroid, p=2, dim=0)
            # Đảm bảo danh sách centroids đủ dài
            while len(self.class_centroids) <= global_idx:
                self.class_centroids.append(None)
            self.class_centroids[global_idx] = centroid

    # ------------------------------------------------------------------
    # Predict: Cosine Classifier — không cần task_id
    # ------------------------------------------------------------------
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Dự đoán global class label bằng Cosine Similarity.

        Thay vì raw logit = z @ W (bị lệch scale giữa các task),
        ta L2-normalize cả z lẫn từng cột W trước khi tính similarity.

        Args:
            z : (N, D) hoặc (D, N) — projected features (dense tensor).
                Nếu là (D, N) sẽ được transpose tự động.

        Returns:
            predicts : (N,) — global class index dự đoán.
        """
        if z.dim() == 2 and z.shape[0] == self._current_dim and z.shape[1] != self._current_dim:
            z = z.T  # (D, N) -> (N, D)

        # L2-normalize feature vectors: z_norm shape (N, D)
        z_norm = F.normalize(z.float(), p=2, dim=1)

        # Lấy phần W có nghĩa: (current_dim, C_seen)
        W_active = self.W[:, :self.num_seen_classes]         # (D, C_seen)
        # L2-normalize từng cột prototype: (D, C_seen)
        W_norm = F.normalize(W_active.float(), p=2, dim=0)

        # Cosine similarity: (N, C_seen)
        cosine_logits = z_norm @ W_norm

        # argmax → global class index
        predicts = cosine_logits.argmax(dim=1)               # (N,)
        return predicts

    # ------------------------------------------------------------------
    # Evaluate: wrapper tiện lợi trả về accuracy
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_embeddings: torch.Tensor,   # (D, N) projected features
        test_labels: torch.Tensor,       # (N,) global labels
    ) -> float:
        """
        Tính test accuracy (%) trên tập test mà không cần task_id.

        Args:
            test_embeddings : (D, N) — projected & WTA features (có thể sparse).
            test_labels     : (N,)   — global class labels.

        Returns:
            accuracy : float — accuracy (%) trên tập test.
        """
        if test_embeddings.is_sparse:
            test_embeddings = test_embeddings.to_dense()  # (D, N)
        predicts = self.predict(test_embeddings)          # (N,)
        accuracy = (predicts.cpu() == test_labels.cpu()).float().mean().item() * 100.0
        return accuracy


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

    # TODO: RAPID - [Ridge Regression State Init]
    # Q và G là các ma trận tích lũy (sufficient statistics) cho Ridge Regression.
    # Q = X^T @ Y (cross-covariance), G = X^T @ X (gram matrix).
    # Trong RAPID, khi expand feature space, kích thước của Q và G cần được
    # cập nhật động (grow) để phù hợp với expand_dim mới sau mỗi task.
    Q = torch.zeros(args.expand_dim, args.num_classes).to(device)
    G = torch.zeros(args.expand_dim, args.expand_dim).to(device)
    last_ridge = None

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
        # Xác định global class indices của task hiện tại
        # (giả định CIL split đồng đều: task t sở hữu class [t*K, (t+1)*K) )
        # ----------------------------------------------------------------
        task_class_start = task * num_classes_per_task
        task_class_end   = task_class_start + num_classes_per_task
        task_class_indices = list(range(task_class_start, task_class_end))

        # ----------------------------------------------------------------
        # One-hot chỉ trên K class của task này (local scope cho Ridge)
        # ----------------------------------------------------------------
        Y_local = target2onehot(train_labels - task_class_start, num_classes_per_task)

        # TODO: RAPID - [Ridge Regression Train (Incremental Update)]
        # Đây là nơi huấn luyện Ridge Regression theo phương pháp recursive/incremental.
        # Q và G được cập nhật cộng dồn (sufficient statistics) qua các task.
        # Wo = (G + ridge*I)^{-1} @ Q là nghiệm closed-form của Ridge Regression.
        # Trong RAPID, khi expand feature space (thêm neuron mới), cần:
        #   1. Pad Q và G cho phù hợp với kích thước mới.
        #   2. Chỉ cập nhật các block tương ứng với neuron mới (block-incremental update).
        # Kích thước hiện tại của feature space (current_proj_dim = t * expand_dim)
        D = current_proj_dim
        Q_task = torch.zeros(D, num_classes_per_task, device=device)
        G_task = torch.zeros(D, D, device=device)
        Q_task = Q_task + train_embeddings @ Y_local
        G_task = G_task + train_embeddings @ train_embeddings.T
        # Tích lũy sufficient statistics toàn cục (grow Q, G nếu cần)
        if Q.shape[0] < D:
            # Grow Q và G theo chiều feature khi projection_matrix mở rộng
            Q = torch.cat([Q, torch.zeros(D - Q.shape[0], Q.shape[1], device=device)], dim=0)
            G = torch.cat(
                [torch.cat([G, torch.zeros(G.shape[0], D - G.shape[1], device=device)], dim=1),
                 torch.zeros(D - G.shape[0], D, device=device)], dim=0
            )
        Q[:, task_class_start:task_class_end] += Q_task
        G[:D, :D] += G_task
        ridge = select_ridge_parameter(train_embeddings.T, Y_local, args.ridge_lower, args.ridge_upper)
        L = torch.linalg.cholesky(G_task + ridge * torch.eye(D, device=device))  # 40% faster
        Wo_task = torch.cholesky_solve(Q_task, L)  # (D, K) — nghiệm Ridge trên D chiều
        training_end = time.time()
        training_time.append(training_end - training_start)

        # TODO: RAPID - [Unified Classifier: Expand & Fit]
        # Expand W để chứa class mới, sau đó ghi nghiệm Ridge vào đúng cột.
        classifier.expand(num_new_classes=num_classes_per_task)
        classifier.fit(
            Wo_task=Wo_task,
            task_class_indices=task_class_indices,
            train_embeddings=train_embeddings,   # (expand_dim, N)
            train_labels=train_labels,           # global labels
        )

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