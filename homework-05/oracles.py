import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

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
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        Ax = self.matvec_Ax(x)
        z = self.b * Ax
        log_loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return log_loss + reg

    def grad(self, x):
        # TODO: Implement
        Ax = self.matvec_Ax(x)
        z = self.b * Ax
        sigma = expit(-z)
        m = len(self.b)
        tmp = -self.b * sigma / m
        grad_loss = self.matvec_ATx(tmp)
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        # TODO: Implement
        Ax = self.matvec_Ax(x)
        z = self.b * Ax
        sigma_z = expit(z)
        sigma_neg_z = expit(-z)
        m = len(self.b)
        s = sigma_z * sigma_neg_z / m
        hess_loss = self.matmat_ATsA(s)
        n = len(x)
        if scipy.sparse.issparse(hess_loss):
            hess_reg = scipy.sparse.diags([self.regcoef] * n)
            return hess_loss + hess_reg
        else:
            hess_reg = self.regcoef * np.eye(n)
            return hess_loss + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        Ax_alpha = Ax + alpha * Ad
        x_alpha = x + alpha * d
        z = self.b * Ax_alpha
        log_loss = np.mean(np.logaddexp(0, -z))
        reg = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        return log_loss + reg

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        Ax_alpha = Ax + alpha * Ad
        x_alpha = x + alpha * d
        z = self.b * Ax_alpha
        sigma = expit(-z)
        m = len(self.b)
        tmp = -self.b * sigma / m
        grad_loss_dot_d = np.dot(self.matvec_ATx(tmp), d)
        grad_reg_dot_d = self.regcoef * np.dot(x_alpha, d)
        return grad_loss_dot_d + grad_reg_dot_d


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """

    def matvec_Ax(x):
        # TODO: Implement
        return A.dot(x)
    
    def matvec_ATx(x):
        # TODO: Implement
        return A.T.dot(x)
      
    def matmat_ATsA(s):
        # TODO: Implement
        if scipy.sparse.issparse(A):
            sA = s[:, np.newaxis] * A
            return A.T.dot(sA)
        else:
            return A.T.dot(s[:, np.newaxis] * A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
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
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
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
