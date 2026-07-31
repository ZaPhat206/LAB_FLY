import re

def patch_main():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update __init__
    init_old = "        self.Wo = None"
    init_new = "        self.Wo = None\n        self.class_counts = {}"
    content = content.replace(init_old, init_new)

    # 2. Update update()
    upd_old = "        # 1. Update prototypes (K-means cho NCM, hoặc mean bình thường)"
    upd_new = """        # Cập nhật số lượng mẫu cho mỗi class
        for c in actual_classes:
            self.class_counts[c] = self.class_counts.get(c, 0) + (lbl == c).sum().item()
            
        # 1. Update prototypes (K-means cho NCM, hoặc mean bình thường)"""
    content = content.replace(upd_old, upd_new)

    # 3. Update ACP block in align_prototypes()
    acp_old = """            elif self.use_acp and num_classes >= 2:
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
                alpha = self.acp_alpha
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
                print(f"    [ACP Subspace] C={num_classes:3d}, r={r}, lam_P={lam_P}, alpha={alpha}")
                G_P = H_proj.T @ H_proj + lam_P * torch.eye(r, device=self.device)
                Q_P = H_proj.T @ Y_target
                L_P = torch.linalg.cholesky(G_P)
                self.P_acp = torch.cholesky_solve(Q_P, L_P)"""
    
    acp_new = """            elif self.use_acp and num_classes >= 2:
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
                
                Q_proj_exact = self.V_r.T @ self.Q_global # (r, C)
                Q_P = Q_proj_exact @ P_target             # (r, r)
                
                lam_P = 0.1
                print(f"    [ACP Subspace] C={num_classes:3d}, r={r}, lam_P={lam_P}, alpha={alpha}")
                G_P = G_proj_exact + lam_P * torch.eye(r, device=self.device)
                L_P = torch.linalg.cholesky(G_P)
                self.P_acp = torch.cholesky_solve(Q_P, L_P)"""
    
    if acp_old not in content:
        print("Warning: ACP old block not found. Checking if it matches somewhat...")
        # Check using regex or just print the block
    else:
        content = content.replace(acp_old, acp_new)

    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_main()
    print("Patch applied")
