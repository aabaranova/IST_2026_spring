import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    def func(self, x):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        Ax = self.matvec_Ax(x)
        z = self.b * Ax
        log_loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return log_loss + reg

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        z = self.b * Ax
        sigma = expit(-z)
        tmp = -self.b * sigma / self.m
        grad_loss = self.matvec_ATx(tmp)
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        z = self.b * Ax
        sigma_z = expit(z)
        sigma_neg_z = expit(-z)
        s = sigma_z * sigma_neg_z / self.m
        hess_loss = self.matmat_ATsA(s)
        n = len(x)
        if scipy.sparse.issparse(hess_loss):
            hess_loss = hess_loss.toarray()
        hess_reg = self.regcoef * np.eye(n)
        return hess_loss + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        Ax_alpha = Ax + alpha * Ad
        x_alpha = x + alpha * d
        z = self.b * Ax_alpha
        log_loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        return log_loss + reg

    def grad_directional(self, x, d, alpha):
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        Ax_alpha = Ax + alpha * Ad
        x_alpha = x + alpha * d
        z = self.b * Ax_alpha
        sigma = expit(-z)
        tmp = -self.b * sigma / self.m
        grad_loss_dot_d = np.dot(self.matvec_ATx(tmp), d)
        grad_reg_dot_d = self.regcoef * np.dot(x_alpha, d)
        return grad_loss_dot_d + grad_reg_dot_d


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    def matvec_Ax(x):
        return A.dot(x)
    
    def matvec_ATx(x):
        return A.T.dot(x)

    def matmat_ATsA(s):
        if scipy.sparse.issparse(A):
            sA = A.multiply(s[:, np.newaxis])
            return A.T.dot(sA)
        else:
            return A.T.dot(s[:, np.newaxis] * A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    n = len(x)
    grad = np.zeros(n)
    f0 = func(x)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_plus = func(x + e_i)
        grad[i] = (f_plus - f0) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    f0 = func(x)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        f_i = func(x + e_i)
        for j in range(i, n):
            e_j = np.zeros(n)
            e_j[j] = eps
            f_j = func(x + e_j)
            f_ij = func(x + e_i + e_j)
            hess[i, j] = (f_ij - f_i - f_j + f0) / (eps ** 2)
            hess[j, i] = hess[i, j]
    return hess

# Переопределяем OptimizedOracle с правильным кэшированием
class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._cached_x = None
        self._cached_Ax = None
        self._cached_d = None
        self._cached_Ad = None
        self._cached_x_alpha = None
        self._cached_Ax_alpha = None

    def _get_Ax(self, x):
        if self._cached_x is not None and np.array_equal(self._cached_x, x):
            return self._cached_Ax
        self._cached_x = x.copy()
        self._cached_Ax = self.matvec_Ax(x)
        return self._cached_Ax

    def _get_Ad(self, d):
        if self._cached_d is not None and np.array_equal(self._cached_d, d):
            return self._cached_Ad
        self._cached_d = d.copy()
        self._cached_Ad = self.matvec_Ax(d)
        return self._cached_Ad

    def func(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        log_loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return log_loss + reg

    def grad(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        sigma = expit(-z)
        tmp = -self.b * sigma / self.m
        grad_loss = self.matvec_ATx(tmp)
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        Ax = self._get_Ax(x)
        z = self.b * Ax
        sigma_z = expit(z)
        sigma_neg_z = expit(-z)
        s = sigma_z * sigma_neg_z / self.m
        hess_loss = self.matmat_ATsA(s)
        n = len(x)
        if scipy.sparse.issparse(hess_loss):
            hess_loss = hess_loss.toarray()
        hess_reg = self.regcoef * np.eye(n)
        return hess_loss + hess_reg

    def func_directional(self, x, d, alpha):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(d)
        Ax_alpha = Ax + alpha * Ad
        x_alpha = x + alpha * d
        self._cached_x_alpha = x_alpha.copy()
        self._cached_Ax_alpha = Ax_alpha
        z = self.b * Ax_alpha
        log_loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        return log_loss + reg

    def grad_directional(self, x, d, alpha):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(d)
        Ax_alpha = Ax + alpha * Ad
        x_alpha = x + alpha * d
        self._cached_x_alpha = x_alpha.copy()
        self._cached_Ax_alpha = Ax_alpha
        z = self.b * Ax_alpha
        sigma = expit(-z)
        tmp = -self.b * sigma / self.m
        grad_loss_dot_d = np.dot(self.matvec_ATx(tmp), d)
        grad_reg_dot_d = self.regcoef * np.dot(x_alpha, d)
        return grad_loss_dot_d + grad_reg_dot_d
