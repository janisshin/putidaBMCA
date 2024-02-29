import numpy as np
import scipy as sp

from scipy.linalg.misc import LinAlgError
from scipy.linalg.lapack import get_lapack_funcs

from aesara import tensor
from aesara.tensor.slinalg import Solve

class SymPosSolve(Solve):   
    """
    Class to allow `solve` to accept a symmetric matrix
    """

    def perform(self, node, inputs, output_storage):
        A, b = inputs
        rval = sympos_solve_wrapper(A, b)
        output_storage[0][0] = rval


sympos_solve = SymPosSolve()#A_structure='symmetric')

class RegularizedSolve(Solve):
    """
    Solve a system of linear equations, Ax = b, while minimizing the norm of x.
    Applies tikhovov regularization.

    """

    __props__ = ('lambda_', *Solve.__props__)

    def __init__(self, lambda_=None):

        Solve.__init__(self)
        self.lambda_ = lambda_ if lambda_ is not None else 0.

    def __repr__(self):
        return 'RegularizedSolve{%s}' % str(self._props())

    def perform(self, node, inputs, output_storage):
        A, b = inputs

        A_hat = A.T @ A + self.lambda_ * np.eye(A.shape[1])
        b_hat = A.T @ b

        rval = sympos_solve_wrapper(A_hat, b_hat)
        output_storage[0][0] = rval

    def L_op(self, inputs, outputs, output_gradients):
        """
        Reverse-mode gradient updates for matrix solve operation.

        """

        A, b = inputs
        c = outputs[0]
        c_bar = output_gradients[0]

        A_hat = A.T.dot(A) + self.lambda_ * tensor.eye(A.shape[1])
        x = sympos_solve(A_hat, c_bar)

        b_bar = A.dot(x)

        def force_outer(l, r):
            return tensor.outer(l, r) if r.ndim == 1 else l.dot(r.T)

        A_bar = force_outer(b - A.dot(c), x) - force_outer(b_bar, c)
        return [A_bar, b_bar]


class LeastSquaresSolve(Solve):
    """
    Solve a system of linear equations, Ax = b, while minimizing the norm of x.

    """

    __props__ = ('driver', *Solve.__props__)

    def __init__(self, driver='gelsy'):

        self.driver = driver
        Solve.__init__(self)

    def __repr__(self):
        return 'LeastSquaresSolve{%s}' % str(self._props())

    def perform(self, node, inputs, output_storage):
        A, b = inputs
        output_storage[0][0] = lstsq_wrapper(A, b, driver=self.driver)

    def L_op(self, inputs, outputs, output_gradients):
        """
        Reverse-mode gradient updates for matrix solve operation.

        """

        A, b = inputs
        c = outputs[0]
        c_bar = output_gradients[0]

        A_hat = A.T.dot(A)
        x = self(A_hat, c_bar)

        b_bar = A.dot(x)

        def force_outer(l, r):
            return tensor.outer(l, r) if r.ndim == 1 else l.dot(r.T)

        A_bar = force_outer(b - A.dot(c), x) - force_outer(b_bar, c)
        return [A_bar, b_bar]


def sympos_solve_wrapper(A, b):
        posv, = get_lapack_funcs(('posv',), (A, b))
        c, rval, info = posv(A, b, lower=False,
                             overwrite_a=False,
                             overwrite_b=False)
    
        if info > 0:
            raise LinAlgError("singular matrix")

        return rval


def lstsq_wrapper(A, b, driver='gelsy'):
    """ Wrap sp.linalg.lstsq to also support the faster _gels solver """
        
    x, _, _, _ = sp.linalg.lstsq(A, b, check_finite=True, lapack_driver=driver)

    assert np.isfinite(x).all()

    return x
