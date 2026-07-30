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
            
            # GCV trace formula for H_hat
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
