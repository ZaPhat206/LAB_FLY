import sys

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 1. Update argparse
    insert_idx = -1
    for i, line in enumerate(lines):
        if "parser.add_argument('--use_procrustes'" in line:
            insert_idx = i - 1
            break
            
    argparse_code = """
    parser.add_argument('--classifier_type', type=str, default='ncm',
                        choices=['ncm', 'ridge_subspace', 'ridge'],
                        help='Type of classifier: ncm, ridge_subspace, ridge')
    parser.add_argument('--use_subspace', action='store_true', default=False,
                        help='Enable subspace extraction for ridge_subspace')
"""
    if insert_idx != -1:
        lines.insert(insert_idx, argparse_code)
    
    # 2. Replace PrototypeClassifier with UnifiedClassifier
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if "class PrototypeClassifier:" in line:
            start_idx = i
        if start_idx != -1 and "def evaluate" in line:
            end_idx = i + 5 # evaluate is 5 lines long
            break
            
    unified_code = """
def select_ridge_parameter(H, Y, lambdas=None):
    if lambdas is None:
        lambdas = [10**i for i in range(-5, 6)]
    N = H.shape[0]
    best_lambda = lambdas[0]
    best_score = float('inf')
    
    HTH = H.T @ H
    HTY = H.T @ Y
    
    for l in lambdas:
        G = HTH + l * torch.eye(H.shape[1], device=H.device)
        try:
            L = torch.linalg.cholesky(G)
            Wo = torch.cholesky_solve(HTY, L)
            Y_pred = H @ Wo
            error = torch.norm(Y - Y_pred, p='fro') ** 2
            
            G_inv = torch.cholesky_inverse(L)
            tr_H_hat = H.shape[1] - l * torch.trace(G_inv)
            
            denominator = (1 - tr_H_hat / N) ** 2
            gcv_score = error / (N * denominator)
            if gcv_score < best_score:
                best_score = gcv_score
                best_lambda = l
        except Exception:
            continue
    return best_lambda

class UnifiedClassifier:
    def __init__(self, device: torch.device, args):
        self.device = device
        self.args = args
        self.classifier_type = args.classifier_type
        self.use_procrustes = args.use_procrustes
        self.num_prototypes = args.num_prototypes
        self.use_subspace = args.use_subspace
        
        if self.classifier_type == 'ridge_subspace' and not self.use_subspace:
            print("Warning: ridge_subspace selected but --use_subspace is False. Falling back to ridge.")
            self.classifier_type = 'ridge'
            
        self.prototypes = {}
        self.etf_prototypes = {}
        self.R = None
        self.V_r = None
        self._current_dim = 0
        
        # Cho Ridge
        self.G_global = None
        self.Q_global = None
        self.Wo = None
        self.last_H = None
        self.last_Y = None

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
        from utils import target2onehot
        self.last_Y = target2onehot(lbl, self.args.num_classes).float().to(self.device)
        
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
                best_lam = 0.1
                G_reg = self.G_global + best_lam * torch.eye(self.G_global.shape[0], device=self.device)
                L = torch.linalg.cholesky(G_reg)
                self.Wo = torch.cholesky_solve(self.Q_global, L)
                return
                
            # Tạo M từ mean của các class
            M_raw = torch.cat([self.prototypes[c] for c in sorted_cls], dim=0) # (C, D)
            
            # SVD: U, S, V = torch.svd(M)
            U, S, Vh = torch.linalg.svd(M_raw, full_matrices=False)
            r = min(num_classes - 1, (S > 1e-6).sum().item())
            if r == 0: r = 1
            
            self.V_r = Vh[:r, :].T # (D, r)
            
            if self.use_procrustes and num_classes >= 3:
                P = create_etf_prototypes(num_classes, r).to(self.device)
                M_proj = M_raw @ self.V_r
                M_proj_norm = F.normalize(M_proj, p=2, dim=1)
                self.R = procrustes_alignment(M_proj_norm, P)
            else:
                self.R = None
                
            # P_proj
            if self.R is not None:
                P_proj = self.V_r @ self.R
            else:
                P_proj = self.V_r
                
            # Vi V_r và R thay đổi mỗi task, Q và G phải được cộng dồn ở không gian gốc 10000D, 
            # sau đó chiếu xuống không gian con mới bằng P_proj để đảm bảo tính đúng đắn toán học.
            G_proj = P_proj.T @ self.G_global @ P_proj # (r, r)
            Q_proj = P_proj.T @ self.Q_global # (r, C)
            
            # GCV trên H_aligned của task hiện tại
            H_aligned = self.last_H @ P_proj
            best_lam = select_ridge_parameter(H_aligned, self.last_Y)
            
            print(f"    [Ridge Subspace] C={num_classes:3d}, r={r}, best_lam={best_lam}")
            
            G_reg = G_proj + best_lam * torch.eye(G_proj.shape[0], device=self.device)
            L = torch.linalg.cholesky(G_reg)
            self.Wo = torch.cholesky_solve(Q_proj, L)
            
        elif self.classifier_type == 'ridge':
            # Giải Ridge gốc 10000D
            best_lam = 0.1 # Tránh GCV trên 10000D vì quá chậm
            print(f"    [Ridge] C={num_classes:3d}, Full D={self._current_dim}")
            G_reg = self.G_global + best_lam * torch.eye(self.G_global.shape[0], device=self.device)
            try:
                L = torch.linalg.cholesky(G_reg)
                self.Wo = torch.cholesky_solve(self.Q_global, L)
            except:
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
                else:
                    H_test_aligned = H_test_proj
                logits = H_test_aligned @ self.Wo
            else:
                logits = z_float @ self.Wo
            return logits.argmax(dim=1).cpu()
            
        elif self.classifier_type == 'ridge':
            logits = z_float @ self.Wo
            return logits.argmax(dim=1).cpu()

    def evaluate(self, H: torch.Tensor, labels: torch.Tensor) -> float:
        if H.is_sparse:
            H = H.to_dense()
        preds = self.predict(H)
        return (preds == labels.cpu()).float().mean().item() * 100.0
"""
    
    if start_idx != -1 and end_idx != -1:
        new_lines = lines[:start_idx] + [unified_code] + lines[end_idx:]
    else:
        print("Could not find PrototypeClassifier")
        return
    
    # 3. Update the instantiation in main()
    inst_idx = -1
    for i, line in enumerate(new_lines):
        if "classifier = PrototypeClassifier(" in line:
            inst_idx = i
            break
            
    if inst_idx != -1:
        inst_code = "    classifier = UnifiedClassifier(device=device, args=args)\n"
        new_lines[inst_idx] = inst_code
        new_lines[inst_idx+1] = ""
        new_lines[inst_idx+2] = ""
        new_lines[inst_idx+3] = ""
        new_lines[inst_idx+4] = ""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Done patching.")

if __name__ == '__main__':
    patch_file('main.py')
