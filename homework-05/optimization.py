import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from scipy.optimize.linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
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
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        if self._method == 'Constant':
            return self.c
        
        elif self._method == 'Armijo':
            # Set starting alpha
            if previous_alpha is not None:
                alpha = previous_alpha
            else:
                alpha = self.alpha_0
            
            # Compute phi(0) and phi'(0)
            phi_0 = oracle.func_directional(x_k, d_k, 0)
            phi_prime_0 = oracle.grad_directional(x_k, d_k, 0)
            
            # Backtracking
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                    return alpha
                alpha = alpha / 2.0
                if alpha < 1e-16:
                    return 0.0
        
        elif self._method == 'Wolfe':
            # Use scipy's Wolfe line search
            phi = lambda a: oracle.func_directional(x_k, d_k, a)
            phi_prime = lambda a: oracle.grad_directional(x_k, d_k, a)
            
            # Try Wolfe conditions
            alpha, _, _, _, _ = scalar_search_wolfe2(phi, phi_prime, phi_0=phi(0), 
                                                      derphi_0=phi_prime(0), 
                                                      c1=self.c1, c2=self.c2)
            
            if alpha is not None:
                return alpha
            
            # Fallback to Armijo backtracking
            if previous_alpha is not None:
                alpha = previous_alpha
            else:
                alpha = self.alpha_0
            
            phi_0 = phi(0)
            phi_prime_0 = phi_prime(0)
            
            while True:
                phi_alpha = phi(alpha)
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
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.
    # Initial gradient norm
    grad_norm = np.linalg.norm(oracle.grad(x_k))
    grad_norm_0 = grad_norm
    
    start_time = datetime.now()
    previous_alpha = None
    
    for iteration in range(max_iter):
        # Check stopping criterion
        if grad_norm**2 <= tolerance * grad_norm_0**2:
            if trace:
                history['time'].append((datetime.now() - start_time).total_seconds())
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(grad_norm)
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            return x_k, 'success', history
        
        # Compute gradient
        grad = oracle.grad(x_k)
        
        # Search direction (negative gradient)
        d_k = -grad
        
        # Line search
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            return x_k, 'computational_error', history
        
        # Update point
        x_k = x_k + alpha * d_k
        
        # Update gradient norm for next iteration
        grad_norm = np.linalg.norm(oracle.grad(x_k))
        previous_alpha = alpha
        
        # Record history
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            print(f"Iteration {iteration}: f(x) = {oracle.func(x_k):.6e}, ||grad|| = {grad_norm:.6e}")
    
    # Max iterations exceeded
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    # Initial gradient norm
    grad_norm = np.linalg.norm(oracle.grad(x_k))
    grad_norm_0 = grad_norm
    
    start_time = datetime.now()
    
    for iteration in range(max_iter):
        # Check stopping criterion
        if grad_norm**2 <= tolerance * grad_norm_0**2:
            if trace:
                history['time'].append((datetime.now() - start_time).total_seconds())
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(grad_norm)
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            return x_k, 'success', history
        
        # Compute gradient and Hessian
        grad = oracle.grad(x_k)
        hess = oracle.hess(x_k)
        
        # Solve Newton system: H * d = -grad
        try:
            # Use Cholesky decomposition for positive definite Hessian
            # Check if Hessian is positive definite
            try:
                # Attempt Cholesky decomposition
                cho_factor = scipy.linalg.cho_factor(hess, lower=True)
                d_k = scipy.linalg.cho_solve(cho_factor, -grad)
            except LinAlgError:
                # If Cholesky fails, fall back to general solve
                d_k = np.linalg.solve(hess, -grad)
        except LinAlgError:
            return x_k, 'newton_direction_error', history
        
        # Check for computational errors
        if np.any(np.isnan(d_k)) or np.any(np.isinf(d_k)):
            return x_k, 'computational_error', history
        
        # Line search (start with alpha=1 for Newton)
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        
        if alpha is None or np.isnan(alpha) or np.isinf(alpha):
            return x_k, 'computational_error', history
        
        # Update point
        x_k = x_k + alpha * d_k
        
        # Update gradient norm for next iteration
        grad_norm = np.linalg.norm(oracle.grad(x_k))
        
        # Record history
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        if display:
            print(f"Iteration {iteration}: f(x) = {oracle.func(x_k):.6e}, ||grad|| = {grad_norm:.6e}")
    
    # Max iterations exceeded
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(grad_norm)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    return x_k, 'success', history
