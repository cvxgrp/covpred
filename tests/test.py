from covpred import Whitener, SMAWhitener, EWMAWhitener, RegressionWhitener, IteratedWhitener, ConstantWhitener
import numpy as np
import cvxpy as cp
from scipy.stats import multivariate_normal

def test_sma_whitener():
    np.random.seed(0)
    Y = np.random.randn(100, 5)
    whitener = SMAWhitener(10)
    Sigmas, Ls, Ywhiten, ts = whitener.whiten(Y)

    for i, t in enumerate(ts):
        Sigma_first = Y[t-10:t].T @ Y[t-10:t] / 10
        np.testing.assert_allclose(
            Sigmas[i],
            Sigma_first
        )
        np.testing.assert_allclose(
            np.linalg.cholesky(np.linalg.inv(Sigma_first)),
            Ls[i]
        )
        np.testing.assert_allclose(
            Ls[i].T @ Y[t],
            Ywhiten[i]
        )

def test_ewma_whitener():
    np.random.seed(0)
    Y = np.random.randn(20, 5)
    halflife = 30
    gamma = np.exp(-np.log(2) / halflife)
    burnin = 10
    whitener = EWMAWhitener(halflife, burnin)
    Sigmas, Ls, Ywhiten, ts = whitener.whiten(Y)

    np.testing.assert_allclose(gamma, whitener.gamma)

    for i, t in enumerate(ts):
        S = sum([np.outer(Y[tau], Y[tau]) * gamma**(t-tau) for tau in range(t)])
        alpha = sum([gamma**(t-tau) for tau in range(t)])
        Sigma_first = S / alpha
        np.testing.assert_allclose(
            Sigmas[i],
            Sigma_first
        )
        np.testing.assert_allclose(
            np.linalg.cholesky(np.linalg.inv(Sigma_first)),
            Ls[i]
        )
        np.testing.assert_allclose(
            Ls[i].T @ Y[t],
            Ywhiten[i]
        )

def test_regression_whitener():
    np.random.seed(0)
    T, n, p = 50, 5, 9
    X = np.random.randn(T, p)
    X = np.maximum(X, -1)
    X = np.minimum(X, 1)
    Y = np.random.randn(T, n)
    diag_rows, diag_cols = np.diag_indices(n)
    off_diag_cols, off_diag_rows = np.triu_indices(n, k=1)
    k = off_diag_rows.size
    epsilon = 1e-6
    lam_1 = .1
    lam_2 = .5
    whitener = RegressionWhitener(epsilon=epsilon, lam_1=lam_1, lam_2=lam_2)
    whitener.fit(Y, X)
    Sigmas, Ls, Ywhiten, ts = whitener.whiten(Y, X)

    A = cp.Variable((n, p))
    b = cp.Variable(n)
    C = cp.Variable((k, p))
    d = cp.Variable(k)

    L = [cp.Variable((n, n)) for _ in range(T)]

    objective = lam_1 / 2 * cp.sum_squares(A) + lam_1 / 2 * cp.sum_squares(C) + \
            lam_2 / 2 * cp.sum_squares(b - 1) + lam_2 / 2 * cp.sum_squares(d)
    constraints = [cp.norm1(A[i,:]) <= b[i] - epsilon for i in range(n)]
    for t in range(T):
        constraints.append(L[t][diag_rows, diag_cols] == A @ X[t] + b)
        constraints.append(L[t][off_diag_rows, off_diag_cols] == C @ X[t] + d)
        constraints.append(L[t].T[off_diag_rows, off_diag_cols] == 0.)
        objective += (-cp.sum(cp.log(cp.diag(L[t]))) + .5 * cp.sum_squares(L[t].T @ Y[t])) / T

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver=cp.MOSEK)

    atol = 1e-3
    np.testing.assert_allclose(A.value, whitener.A, atol=atol)
    np.testing.assert_allclose(b.value, whitener.b, atol=atol)
    np.testing.assert_allclose(C.value, whitener.C, atol=atol)
    np.testing.assert_allclose(d.value, whitener.d, atol=atol)

def test_iterated_whitener():
    np.random.seed(0)
    T, n, p = 200, 5, 9
    X = np.random.randn(T, p)
    Y = np.random.randn(T, n)

    whitener1 = SMAWhitener(20)
    whitener2 = EWMAWhitener(20, 20)

    iterated_whitener = IteratedWhitener([whitener1, whitener2])
    Sigmas, Ls, Ywhiten, ts = iterated_whitener.whiten(Y, X)

    Sigmas1, Ls1, Ywhiten1, ts1 = whitener1.whiten(Y, X)
    Sigmas2, Ls2, Ywhiten2, ts2 = whitener2.whiten(Ywhiten1, X[ts1])

    Ls_1 = np.array([Ls1[t] @ Ls2[i] for i, t in enumerate(ts2)])
    Sigmas_1 = [np.linalg.inv(L @ L.T) for L in Ls_1]
    ts_1 = ts1[ts2]
    Ywhiten_1 = Ywhiten2

    np.testing.assert_allclose(Ls, Ls_1)
    np.testing.assert_allclose(Sigmas, Sigmas_1)
    np.testing.assert_allclose(ts, ts_1)
    np.testing.assert_allclose(Ywhiten, Ywhiten_1)


def test_score():
    Y = np.random.randn(20, 5)
    Ytest = np.random.randn(20, 5)
    Sigma = np.cov(Y.T)

    whitener = ConstantWhitener()
    whitener.fit(Y)
    score = whitener.score(Ytest)

    mn = multivariate_normal(np.zeros(5), Sigma)
    np.testing.assert_allclose(score, mn.logpdf(Ytest).mean())

if __name__ == "__main__":
    test_sma_whitener()
    test_ewma_whitener()
    test_regression_whitener()
    test_iterated_whitener()
    test_score()