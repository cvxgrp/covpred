import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class Whitener(object):
    def __init__(self):
        pass

    def whiten(self, Y, X=None):
        return NotImplementedError

    def fit(self, Y, X=None):
        pass

    def score(self, Y, X=None):
        T, n = Y.shape
        Sigmas, _, _, ts = self.whiten(Y, X)
        score = 0.
        for i, t in enumerate(ts):
            score += -n*np.log(2*np.pi) / 2 - np.linalg.slogdet(Sigmas[i])[1] / 2 - Y[t] @ np.linalg.solve(Sigmas[i], Y[t]) / 2
        return score / len(ts)

class ConstantWhitener(Whitener):
    def __init__(self , lam=0):
        self.lam = lam

    def fit(self, Y, X=None):
        self.Sigma = np.cov(Y.T) + self.lam * np.eye(Y.shape[1])
        self.L = np.linalg.cholesky(np.linalg.inv(self.Sigma))

    def whiten(self, Y, X=None):
        T, n = Y.shape
        Ywhiten = Y @ self.L.T
        return np.array([self.Sigma for _ in range(T)]), \
                np.array([self.L for _ in range(T)]), \
                Ywhiten, \
                np.arange(T)

class IgnoreWhitener(Whitener):
    def __init__(self, steps):
        self.steps = steps

    def whiten(self, Y, X=None):
        T, n = Y.shape
        Sigmas = [np.eye(n) for _ in range(T-self.steps)]
        Ls = [np.eye(n) for _ in range(T-self.steps)]
        Ywhiten = Y[self.steps:]
        ts = np.arange(self.steps, T)
        return np.array(Sigmas), np.array(Ls), Ywhiten, ts

class SMAWhitener(Whitener):
    def __init__(self, memory):
        self.memory = memory

    def whiten(self, Y, X=None):
        T, n = Y.shape
        assert self.memory > n, "memory must be at least n"
        Sigma = np.zeros((n, n))
        Sigmas = []
        Ls = []
        ts = []
        Ywhiten = []
        for t in range(T - 1):
            update = np.outer(Y[t], Y[t])
            downdate = np.zeros((n, n))
            if t >= self.memory:
                downdate = np.outer(Y[t-self.memory], Y[t-self.memory])
            Sigma = Sigma + (1 / self.memory) * (update - downdate)
            if t >= self.memory - 1:
                Theta = np.linalg.inv(Sigma)
                L = np.linalg.cholesky(Theta)
                Sigmas.append(Sigma)
                Ls.append(L)
                ts.append(t+1)
                Ywhiten.append(L.T @ Y[t+1])
        return np.array(Sigmas), np.array(Ls), np.array(Ywhiten), np.array(ts)


class EWMAWhitener(Whitener):
    def __init__(self, halflife, burnin=10):
        self.halflife = halflife
        self.gamma = np.exp(-np.log(2) / halflife)
        self.burnin = burnin

    def whiten(self, Y, X=None):
        T, n = Y.shape
        assert self.burnin > n, "burnin must be at least n"
        Sigma = np.zeros((n, n))
        Sigmas = []
        Ls = []
        Ywhiten = []
        ts = []
        alpha_inv = 0
        for t in range(T-1):
            alpha_inv_new = (alpha_inv + 1) * self.gamma
            Sigma = self.gamma / alpha_inv_new * (Sigma * alpha_inv + np.outer(Y[t], Y[t]))
            alpha_inv = alpha_inv_new

            if t >= self.burnin - 1:
                Theta = np.linalg.inv(Sigma)
                L = np.linalg.cholesky(Theta)
                Sigmas.append(Sigma)
                Ls.append(L)
                Ywhiten.append(L.T @ Y[t+1])
                ts.append(t+1)
        return np.array(Sigmas), np.array(Ls), np.array(Ywhiten), np.array(ts)

class PermutationWhitener(Whitener):
    def __init__(self, order):
        n = order.size
        self.P = np.eye(n)[order,:]
        self.order = order

    def whiten(self, Y, X=None):
        T, n = Y.shape
        Sigmas = np.array([np.eye(n) for _ in range(T)])
        Ls = np.array([self.P.T for _ in range(T)])
        Ywhiten = Y @ self.P.T
        ts = np.arange(T)
        return Sigmas, Ls, Ywhiten, ts

    def reverse(self):
        order1 = np.zeros(self.order.size, dtype=np.int)
        for i in range(self.order.size):
            order1[self.order[i]] = i
        return PermutationWhitener(np.array(order1))
    
class RegressionWhitener(Whitener):
    def __init__(self, epsilon=1e-3, lam_1=0, lam_2=0):
        self.epsilon = epsilon
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.fitted = False

    def fit(self, Y, X):
        self.A, self.b, self.C, self.d = self._fit(X, Y, epsilon=self.epsilon, lam_1=self.lam_1, lam_2=self.lam_2)
        self.fitted = True
        
    def whiten(self, Y, X):
        assert self.fitted, "must call fit before whitening"
        T, _ = Y.shape
        Ywhiten = []
        Ls = self._predict(self.A, self.b, self.C, self.d, X)
        Sigmas = []
        for t in range(T):
            Sigmas.append(np.linalg.inv(Ls[t] @ Ls[t].T))
            Ywhiten.append(Ls[t].T @ Y[t])
        return np.array(Sigmas), Ls, np.array(Ywhiten), np.arange(T)
    
    def _fit(self, X, Y, epsilon=1e-6, lam_1=0, lam_2=0, **kwargs):
        assert np.all(X >= -1), "X must be in [-1, 1] and not missing"
        assert np.all(X <= 1), "X must be in [-1, 1] and not missing"
        T, p = X.shape
        T, n = Y.shape
        diag_rows, diag_cols = np.diag_indices(n)
        off_diag_cols, off_diag_rows = np.triu_indices(n, k=1)
        k = off_diag_rows.size
        def f(x):
            Aplus = x[:n*p].reshape(n, p)
            Aneg = x[n*p:n*p*2].reshape(n, p)
            bplus = x[n*p*2:n*(p*2+1)]
            C = x[n*(p*2+1):n*(p*2+1)+k*p].reshape(k, p)
            d = x[n*(p*2+1)+k*p:n*(p*2+1)+k*p+k]
            A = Aplus - Aneg
            b = (Aplus + Aneg) @ np.ones(p) + epsilon + bplus

            L = np.zeros((T, n, n))

            L[:, diag_rows, diag_cols] = X @ A.T + b
            L[:, off_diag_rows, off_diag_cols] = X @ C.T + d

            f = -np.log(L[:, diag_rows, diag_cols]).sum() / T + .5 * np.square((Y[:,:,None] * L).sum(axis=1)).sum() / T + \
                lam_1 / 2 * (np.sum(np.square(A)) + np.sum(np.square(C))) + \
                lam_2 / 2 * (np.sum(np.square(b - 1)) + np.sum(np.square(d)))

            L_grad = np.zeros((T, n, n))
            L_grad[:, diag_rows, diag_cols] = -1.0 / L[:, diag_rows, diag_cols]
            L_grad += Y[:,:,None] * (L.transpose(0,2,1) * Y[:,None,:]).sum(axis=2)[:,None,:]

            Aplus_grad = (L_grad[:, diag_rows, diag_cols][:,:,None] * (X[:,None,:] + 1)).sum(axis=0) / T + \
                    lam_1 * A + lam_2 * np.outer(b - 1, np.ones(p))
            Aneg_grad = (L_grad[:, diag_rows, diag_cols][:,:,None] * (-X[:,None,:] + 1)).sum(axis=0) / T - \
                    lam_1 * A + lam_2 * np.outer(b - 1, np.ones(p))
            C_grad = (L_grad[:, off_diag_rows, off_diag_cols][:,:,None] * X[:,None,:]).sum(axis=0) / T + lam_1 * C

            bplus_grad = L_grad[:, diag_rows, diag_cols].sum(axis=0) / T + lam_2 * (b - 1)
            d_grad = L_grad[:, off_diag_rows, off_diag_cols].sum(axis=0) / T + lam_2 * d

            grad = np.concatenate([
                Aplus_grad.flatten(),
                Aneg_grad.flatten(),
                bplus_grad.flatten(),
                C_grad.flatten(),
                d_grad.flatten()
            ])
            return f, grad
        bounds = [(0, np.inf)] * (n*p) + [(0,np.inf)] * (n*p) + \
                [(0, np.inf)] * n + [(-np.inf, np.inf)] * k * p + [(-np.inf, np.inf)] * k
        x = np.zeros(len(bounds))
        x[2*n*p:2*n*p+n] = 1 - epsilon
        x, fstar, info = fmin_l_bfgs_b(f, x, bounds=bounds, **kwargs)
        Aplus = x[:n*p].reshape(n, p)
        Aneg = x[n*p:n*p*2].reshape(n, p)
        bplus = x[n*p*2:n*(p*2+1)]
        C = x[n*(p*2+1):n*(p*2+1)+k*p].reshape(k, p)
        d = x[n*(p*2+1)+k*p:n*(p*2+1)+k*p+k]
        A = Aplus - Aneg
        b = (Aplus + Aneg) @ np.ones(p) + epsilon + bplus

        return A, b, C, d

    def _predict(self, A, b, C, d, X):
        T, p = X.shape
        n = A.shape[0]
        diag_rows, diag_cols = np.diag_indices(n)
        off_diag_cols, off_diag_rows = np.triu_indices(n, k=1)
        k = off_diag_rows.size
        Ls = np.zeros((T, n, n))
        Ls[:, diag_rows, diag_cols] = X @ A.T + b
        Ls[:, off_diag_rows, off_diag_cols] = X @ C.T + d
        return Ls

class DiagonalWhitener(Whitener):
    def __init__(self, lam):
        self.lam = lam

    def fit(self, Y, X):
        N, n = Y.shape
        N, p = X.shape
        def f(x):
            A = x[:n*p].reshape(n, p)
            b = x[n*p:].reshape(n)
            
            pred = X @ A.T + b
            f = np.sum(pred + np.exp(-pred) * Y**2) / N + self.lam / 2 * np.sum(np.square(A))
            
            A_grad = 1 / N * np.outer(np.ones(n), np.ones(N) @ X) - 1 / N * (np.exp(-pred) * Y**2).T @ X + self.lam * A
            b_grad = np.ones(n) - 1 / N * np.ones(N) @ (np.exp(-pred) * Y**2)
            
            grad = np.append(A_grad.flatten(), b_grad.flatten())
            return f, grad
        x = np.zeros(n*p + n)
        x, fstar, info = fmin_l_bfgs_b(f, x)
        self._A = x[:n*p].reshape(n, p)
        self._b = x[n*p:]

    def whiten(self, Y, X):
        N, n = Y.shape
        N, p = X.shape
        Sigmas = [np.diag(np.exp(self._A @ X[i] + self._b)) for i in range(N)]
        Ls = [np.diag(np.exp((-self._A @ X[i] - self._b) / 2)) for i in range(N)]
        Ys = [Ls[i].T @ Y[i] for i in range(N)]
        ts = np.arange(N)
        return np.array(Sigmas), np.array(Ls), np.array(Ys), ts
    

class IteratedWhitener(Whitener):
    def __init__(self, whiteners):
        """
        We apply the whiteners from left to right.
        """
        self.whiteners = whiteners

    def fit(self, Y, X):
        T, n = Y.shape
        Ls = [np.eye(n) for _ in range(T)]
        ts = np.arange(T)
        for w in self.whiteners:
            w.fit(Y, X)
            _, Ls_new, Y, ts_temp = w.whiten(Y, X)
            ts = ts[ts_temp]
            Ls = [Ls[t] @ Ls_new[i] for i, t in enumerate(ts_temp)]
            if X is not None:
                X = X[ts_temp]

    def whiten(self, Y, X=None):
        T, n = Y.shape
        Ls = [np.eye(n) for _ in range(T)]
        ts = np.arange(T)
        for w in self.whiteners:
            _, Ls_new, Y, ts_temp = w.whiten(Y, X)
            ts = ts[ts_temp]
            Ls = [Ls[t] @ Ls_new[i] for i, t in enumerate(ts_temp)]
            if X is not None:
                X = X[ts_temp]
        Sigmas = [np.linalg.inv(L @ L.T) for L in Ls]
        return np.array(Sigmas), np.array(Ls), Y, ts