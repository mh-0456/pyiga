
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyiga'))
import pyiga
#
import scipy
import itertools

from pyiga import bspline, assemble, vform, geometry, vis, solvers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# assemble matrix
from scipy.sparse import coo_matrix, block_diag, bmat
from scipy.sparse import bsr_matrix

HAVE_MKL = True
try:
    import pyMKL
    HAVE_MKL = True
    print('OK')
except:
    HAVE_MKL = False
    print(' no MKL')

    
    
class PardisoSolverWrapper(scipy.sparse.linalg.LinearOperator):
    """Wraps a PARDISO solver object and frees up the memory when deallocated."""
    def __init__(self, shape, dtype, solver):
        self.solver = solver
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=shape, dtype=dtype)
    def _matvec(self, x):
        return self.solver.solve(x)
    def _matmat(self, x):
        return self.solver.solve(x)
    def __del__(self):
        self.solver.clear()
        self.solver = None
        
        
        
def make_solver(B, symmetric=False, spd=False):
    """Return a :class:`LinearOperator` that acts as a linear solver for the
    (dense or sparse) square matrix `B`.

    If `B` is symmetric, passing ``symmetric=True`` may try to take advantage of this.
    If `B` is symmetric and positive definite, pass ``spd=True``.
    """
    if spd:
        symmetric = True
# Gauß'sche Eliminationsverfahren - LU Zerlegung (auch LR für left-right)
    if scipy.sparse.issparse(B):
        if HAVE_MKL:
             # use MKL Pardiso
            mtype = 11   # real, nonsymmetric
            if symmetric:
                mtype = 2 if spd else -2
            solver = pyMKL.pardisoSolver(B, mtype)
            solver.factor()
            return PardisoSolverWrapper(B.shape, B.dtype, solver)
        else:
            print('use SuperLU')
                # use SuperLU (unless scipy uses UMFPACK?) -- really slow!
            spLU = scipy.sparse.linalg.splu(B.tocsc(), permc_spec='NATURAL')
            M= scipy.sparse.linalg.LinearOperator(B.shape, dtype=B.dtype, matvec=spLU.solve, matmat=spLU.solve)
            return M
# Cholesky Zerlegung: Matrix muss symmetrisch und positiv definit sein!                  
    else:
        if symmetric:
            print('use Cholesky')
            chol = scipy.linalg.cho_factor(B, check_finite=False)
            solve = lambda x: scipy.linalg.cho_solve(chol, x, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape, dtype=B.dtype,
                    matvec=solve, matmat=solve)
        else:
            print('Matrix is not symmetric')
            LU = scipy.linalg.lu_factor(B, check_finite=False)
            solve = lambda x: scipy.linalg.lu_solve(LU, x, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape, dtype=B.dtype,
                    matvec=solve, matmat=solve)