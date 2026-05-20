import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict

# Попробуем импортировать line_search из scipy
try:
    from scipy.optimize import line_search_wolfe2
    USE_SCIPY_WOLFE = True
except ImportError:
    try:
        from scipy.optimize.linesearch import line_search_wolfe2
        USE_SCIPY_WOLFE = True
    except ImportError:
        USE_SCIPY_WOLFE = False


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c
        
        elif self._method == 'Armijo':
            if previous_alpha is not None:
                alpha = previous_alpha
            else:
                alpha = self.alpha_0
            
            phi_0 = oracle.func_directional(x_k, d_k, 0)
            phi_prime_0 = oracle.grad_directional(x_k, d_k, 0)
            
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                    return alpha
                alpha = alpha / 2.0
                if alpha < 1e-16:
                    return 0.0
        
        elif self._method == 'Wolfe':
            if USE_SCIPY_WOLFE:
                phi = lambda a: oracle.func_directional(x_k, d_k, a)
                phi_prime = lambda a: oracle.grad_directional(x_k, d_k, a)
                result = line_search_wolfe2(phi, phi_prime, phi(0), phi_prime(0),
                                            c1=self.c1, c2=self.c2)
                alpha = result[0]
                if alpha is not None:
                    return alpha
            
            # Fallback to Armijo
            if previous_alpha is not None:
                alpha = previous_alpha
            else:
                alpha = self.alpha_0
            
            phi_0 = oracle.func_directional(x_k, d_k, 0)
            phi_prime_0 = oracle.grad_directional(x_k, d_k, 0)
            
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                    return alpha
                alpha = alpha / 2.0
                if alpha < 1e-16:
                    return 0.0
        
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    grad_norm = np.linalg.norm(oracle.grad(x_k))
    grad_norm_0 = grad_norm
    
    if grad_norm_0 == 0:
        if trace:
            history['time'].append(0.0)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(0.0)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        return x_k, 'success', history
    
    start_time = datetime.now()
    previous_alpha = None
    
    for iteration in range(max_iter):
        if grad_norm**2 <= tolerance * grad_norm_0**2:
            if trace:
                history['time'].append((datetime.now() - start_time).total_seconds())
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(grad_norm)
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            return x_k, 'success', history
        
        grad = oracle.grad(x_k)
        d_k = -grad
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            return x_k, 'computational_error', history
        
        x_k = x_k + alpha * d_k
        grad_norm = np.linalg.norm(oracle.grad(x_k))
        previous_alpha = alpha
        
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            if iteration == 0:
                print("Gradient descent started")
            print(f"Iteration {iteration}: f(x) = {oracle.func(x_k):.6e}, ||grad|| = {grad_norm:.6e}")
    
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    grad_norm = np.linalg.norm(oracle.grad(x_k))
    grad_norm_0 = grad_norm
    
    if grad_norm_0 == 0:
        if trace:
            history['time'].append(0.0)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(0.0)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        return x_k, 'success', history
    
    start_time = datetime.now()
    
    for iteration in range(max_iter):
        if grad_norm**2 <= tolerance * grad_norm_0**2:
            if trace:
                history['time'].append((datetime.now() - start_time).total_seconds())
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(grad_norm)
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            return x_k, 'success', history
        
        grad = oracle.grad(x_k)
        hess = oracle.hess(x_k)
        
        try:
            try:
                cho_factor = scipy.linalg.cho_factor(hess, lower=True)
                d_k = scipy.linalg.cho_solve(cho_factor, -grad)
            except LinAlgError:
                d_k = np.linalg.solve(hess, -grad)
        except LinAlgError:
            return x_k, 'computational_error', history
        
        if np.any(np.isnan(d_k)) or np.any(np.isinf(d_k)):
            return x_k, 'computational_error', history
        
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            return x_k, 'computational_error', history
        
        x_k = x_k + alpha * d_k
        grad_norm = np.linalg.norm(oracle.grad(x_k))
        
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            if iteration == 0:
                print("Newton method started")
            print(f"Iteration {iteration}: f(x) = {oracle.func(x_k):.6e}, ||grad|| = {grad_norm:.6e}")
    
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    
    return x_k, 'iterations_exceeded', history
