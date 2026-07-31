import re

def patch_main():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update argparse
    if '--use_acp' not in content:
        arg_patch = """    parser.add_argument('--use_subspace', action='store_true', default=False,
                        help='Enable subspace extraction for ridge_subspace')
    parser.add_argument('--use_acp', action='store_true', default=False,
                        help='Enable Analytic Contrastive Projection on subspace')
    parser.add_argument('--subspace_rank', type=int, default=50,
                        help='Rank r of the subspace (default 50)')"""
        content = content.replace("    parser.add_argument('--use_subspace', action='store_true', default=False,\n                        help='Enable subspace extraction for ridge_subspace')", arg_patch)

    # 2. Update __init__
    init_patch = """        self.use_subspace = args.use_subspace
        self.use_acp = getattr(args, 'use_acp', False)
        self.subspace_rank = getattr(args, 'subspace_rank', 50)"""
    content = content.replace("        self.use_subspace = args.use_subspace", init_patch)

    p_acp_patch = """        self.V_r = None
        self.P_acp = None"""
    content = content.replace("        self.V_r = None", p_acp_patch)

    last_lbl_patch = """        self.last_H = None
        self.last_Y = None
        self.last_lbl = None"""
    content = content.replace("        self.last_H = None\n        self.last_Y = None", last_lbl_patch)

    # 3. Update update()
    upd_patch = """        self.last_H = H_dev
        self.last_lbl = lbl.clone()
        from utils import target2onehot"""
    content = content.replace("        self.last_H = H_dev\n        from utils import target2onehot", upd_patch)

    # 4. Update align_prototypes()
    # Find the block for ridge_subspace
    acp_code = """
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
                H_proj = self.last_H @ self.V_r # (N_curr, r)
                lbl_curr = self.last_lbl
                
                # Tính M_proj cho tất cả các lớp đã thấy (C x r)
                M_proj = M_raw @ self.V_r
                
                # Tính shared covariance từ H_proj của task hiện tại
                H_centered = torch.empty_like(H_proj)
                unique_c = lbl_curr.unique()
                for c in unique_c:
                    mask = (lbl_curr == c)
                    if mask.sum() > 0:
                        H_centered[mask] = H_proj[mask] - H_proj[mask].mean(dim=0)
                
                Sigma = (H_centered.T @ H_centered) / H_centered.shape[0] # (r, r)
                # Đảm bảo tính khả nghịch
                Sigma = Sigma + 1e-6 * torch.eye(r, device=self.device)
                
                Us, Ss, _ = torch.linalg.svd(Sigma)
                Sigma_inv_half = Us @ torch.diag((Ss + 1e-6)**(-0.5)) @ Us.T
                Sigma_half = Us @ torch.diag(Ss**0.5) @ Us.T
                
                M_whitened = M_proj @ Sigma_inv_half # (C, r)
                
                Um, Sm, Vhm = torch.linalg.svd(M_whitened, full_matrices=False)
                # Tăng cường separation
                alpha = 1.0
                S_enhanced = Sm + alpha
                
                # P_target: (C, r)
                P_target = Um @ torch.diag(S_enhanced) @ Vhm @ Sigma_half
                
                # Tạo Y_target (N_curr, r) tương ứng với label của từng mẫu
                # Y_target[i] = P_target[lbl_curr[i]]
                # Lưu ý: P_target được index theo sorted_cls (0..num_classes-1).
                # Label trong lbl_curr có thể là giá trị thực tế (VD: 10, 11,...).
                # Ta cần ánh xạ lbl_curr về chỉ số trong sorted_cls.
                cls_to_idx = {c: i for i, c in enumerate(sorted_cls)}
                idx_curr = torch.tensor([cls_to_idx[c.item()] for c in lbl_curr], device=self.device)
                Y_target = P_target[idx_curr]
                
                # Học phép chiếu P (r x r)
                lam_P = 0.1
                G_P = H_proj.T @ H_proj + lam_P * torch.eye(r, device=self.device)
                Q_P = H_proj.T @ Y_target
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
"""

    old_acp = """
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
                P_proj = self.V_r"""
    content = content.replace(old_acp, acp_code)

    # 5. Update predict
    old_pred = """        elif self.classifier_type == 'ridge_subspace':
            if self.V_r is not None:
                H_test_proj = z_float @ self.V_r
                if self.R is not None:
                    H_test_aligned = H_test_proj @ self.R
                else:
                    H_test_aligned = H_test_proj
                logits = H_test_aligned @ self.Wo"""
    
    new_pred = """        elif self.classifier_type == 'ridge_subspace':
            if self.V_r is not None:
                H_test_proj = z_float @ self.V_r
                if self.R is not None:
                    H_test_aligned = H_test_proj @ self.R
                elif self.P_acp is not None:
                    H_test_aligned = H_test_proj @ self.P_acp
                else:
                    H_test_aligned = H_test_proj
                logits = H_test_aligned @ self.Wo"""
    content = content.replace(old_pred, new_pred)

    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_main()
    print("Done")
