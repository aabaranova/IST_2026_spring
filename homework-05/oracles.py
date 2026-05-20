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
        self._cached_x = None
        self._cached_Ax = None
        self._cached_d = None
        self._cached_Ad = None
        self._cached_tmp = None

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
        self._cached_tmp = tmp
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
        z = self.b * Ax_alpha
        log_loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        return log_loss + reg

    def grad_directional(self, x, d, alpha):
        Ax = self._get_Ax(x)
        Ad = self._get_Ad(d)
        Ax_alpha = Ax + alpha * Ad
        x_alpha = x + alpha * d
        z = self.b * Ax_alpha
        sigma = expit(-z)
        # Используем закэшированный tmp если есть, иначе вычисляем
        if self._cached_tmp is not None:
            tmp = self._cached_tmp
        else:
            tmp = -self.b * sigma / self.m
        grad_loss_dot_d = np.dot(self.matvec_ATx(tmp), d)
        grad_reg_dot_d = self.regcoef * np.dot(x_alpha, d)
        return grad_loss_dot_d + grad_reg_dot_d
