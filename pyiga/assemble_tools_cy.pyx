# cython: language_level=3
# cython: profile=False
# cython: linetrace=False
# cython: binding=False

from builtins import range as range_it   # Python 2 compatibility

cimport cython
from cython.parallel import prange
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

import scipy.sparse

import pyiga
from . import bspline
from .quadrature import make_iterated_quadrature
from .mlmatrix import MLBandedMatrix, get_transpose_idx_for_bidx
from . cimport fast_assemble_cy

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import itertools

################################################################################
# Public utility functions
################################################################################

cdef inline void from_seq3(size_t i, size_t[3] ndofs, size_t[3] out) nogil:
    out[2] = i % ndofs[2]
    i /= ndofs[2]
    out[1] = i % ndofs[1]
    i /= ndofs[1]
    out[0] = i

# returns an array where each row contains:
#  i0 i1 i2  j0 j1 j2  t0 t1 t2
# where ix and jx are block indices of a matrix entry
# and (t0,t1,t2) is a single tile index which is contained
# in the joint support for the matrix entry
def prepare_tile_indices3(ij_arr, meshsupp, numdofs):
    cdef size_t[3] ndofs
    ndofs[:] = numdofs
    cdef vector[unsigned] result
    cdef size_t I[3]
    cdef size_t J[3]
    cdef size_t[:, :] ij = ij_arr
    cdef size_t N = ij.shape[0], M = 0
    cdef IntInterval[3] intvs

    for k in range(N):
        from_seq3(ij[k,0], ndofs, I)
        from_seq3(ij[k,1], ndofs, J)

        for r in range(3):
            ii = I[r]
            jj = J[r]
            intvs[r] = intersect_intervals(
                make_intv(meshsupp[r][ii, 0], meshsupp[r][ii, 1]),
                make_intv(meshsupp[r][jj, 0], meshsupp[r][jj, 1]))

        for t0 in range(intvs[0].a, intvs[0].b):
            for t1 in range(intvs[1].a, intvs[1].b):
                for t2 in range(intvs[2].a, intvs[2].b):
                    result.push_back(I[0])
                    result.push_back(I[1])
                    result.push_back(I[2])
                    result.push_back(J[0])
                    result.push_back(J[1])
                    result.push_back(J[2])
                    result.push_back(t0)
                    result.push_back(t1)
                    result.push_back(t2)
                    M += 1
    return np.array(<unsigned[:result.size()]> result.data(), order='C').reshape((M,9))


# Used to recombine the results of tile-wise assemblers, where a single matrix
# entry is split up into is contributions per tile. This class sums these
# contributions up again.
cdef class MatrixEntryAccumulator:
    cdef ssize_t idx
    cdef int num_indices
    cdef double[::1] _result
    cdef unsigned[16] old_indices

    def __init__(self, int num_indices, size_t N):
        self.idx = -1
        assert num_indices <= 16
        self.num_indices = num_indices
        self._result = np.empty(N)
        for i in range(16):
            self.old_indices[i] = 0xffffffff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint index_changed(self, unsigned[:, :] indices, size_t i) nogil:
        cdef bint changed = False
        for k in range(self.num_indices):
            if indices[i, k] != self.old_indices[k]:
                changed = True
                break
        if changed: # update old_indices
            for k in range(self.num_indices):
                self.old_indices[k] = indices[i, k]
        return changed

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef process(self, unsigned[:, :] indices, double[::1] values):
        cdef size_t M = indices.shape[0]
        for i in range(M):
            if self.index_changed(indices, i):
                self.idx += 1
                self._result[self.idx] = values[i]
            else:
                self._result[self.idx] += values[i]

    @property
    def result(self):
        return self._result[:self.idx+1]


################################################################################
# Internal helper functions
################################################################################

cdef struct IntInterval:
    int a
    int b

cdef IntInterval make_intv(int a, int b) nogil:
    cdef IntInterval intv
    intv.a = a
    intv.b = b
    return intv

cdef IntInterval intersect_intervals(IntInterval intva, IntInterval intvb) nogil:
    return make_intv(max(intva.a, intvb.a), min(intva.b, intvb.b))


cdef int next_lexicographic2(size_t[2] cur, size_t start[2], size_t end[2]) nogil:
    cdef size_t i
    for i in range(2):
        cur[i] += 1
        if cur[i] == end[i]:
            if i == (2-1):
                return 0
            else:
                cur[i] = start[i]
        else:
            return 1

cdef int next_lexicographic3(size_t[3] cur, size_t start[3], size_t end[3]) nogil:
    cdef size_t i
    for i in range(3):
        cur[i] += 1
        if cur[i] == end[i]:
            if i == (3-1):
                return 0
            else:
                cur[i] = start[i]
        else:
            return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef IntInterval find_joint_support_functions(ssize_t[:,::1] meshsupp, long i) nogil:
    cdef long j, n, minj, maxj
    minj = j = i
    while j >= 0 and meshsupp[j,1] > meshsupp[i,0]:
        minj = j
        j -= 1

    maxj = i
    j = i + 1
    n = meshsupp.shape[0]
    while j < n and meshsupp[j,0] < meshsupp[i,1]:
        maxj = j
        j += 1
    return make_intv(minj, maxj+1)


#### determinants and inverses

def det_and_inv(X):
    """Return (np.linalg.det(X), np.linalg.inv(X)), but much
    faster for 2x2- and 3x3-matrices."""
    d = X.shape[-1]
    if d == 2:
        det = np.empty(X.shape[:-2])
        inv = det_and_inv_2x2(X, det)
        return det, inv
    elif d == 3:
        det = np.empty(X.shape[:-2])
        inv = det_and_inv_3x3(X, det)
        return det, inv
    else:
        return np.linalg.det(X), np.linalg.inv(X)

def determinants(X):
    """Compute the determinants of an ndarray of square matrices.

    This behaves mostly identically to np.linalg.det(), but is faster for 2x2 matrices."""
    shape = X.shape
    d = shape[-1]
    assert shape[-2] == d, "Input matrices need to be square"
    if d == 2:
        # optimization for 2x2 matrices
        assert len(shape) == 4, "Only implemented for n x m x 2 x 2 arrays"
        return X[:,:,0,0] * X[:,:,1,1] - X[:,:,0,1] * X[:,:,1,0]
    elif d == 3:
        return determinants_3x3(X)
    else:
        return np.linalg.det(X)

def inverses(X):
    if X.shape[-2:] == (2,2):
        return inverses_2x2(X)
    elif X.shape[-2:] == (3,3):
        return inverses_3x3(X)
    else:
        return np.linalg.inv(X)

#### 2D determinants and inverses

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,::1] det_and_inv_2x2(double[:,:,:,::1] X, double[:,::1] det_out):
    cdef long m,n, i,j
    cdef double det, a,b,c,d
    m,n = X.shape[0], X.shape[1]

    cdef double[:,:,:,::1] Y = np.empty_like(X)
    for i in prange(m, nogil=True, schedule='static'):
        for j in range(n):
            a,b,c,d = X[i,j, 0,0], X[i,j, 0,1], X[i,j, 1,0], X[i,j, 1,1]
            det = a*d - b*c
            det_out[i,j] = det
            Y[i,j, 0,0] =  d / det
            Y[i,j, 0,1] = -b / det
            Y[i,j, 1,0] = -c / det
            Y[i,j, 1,1] =  a / det
    return Y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,::1] inverses_2x2(double[:,:,:,::1] X):
    cdef size_t m,n, i,j
    cdef double det, a,b,c,d
    m,n = X.shape[0], X.shape[1]

    cdef double[:,:,:,::1] Y = np.empty_like(X)
    for i in range(m):
        for j in range(n):
            a,b,c,d = X[i,j, 0,0], X[i,j, 0,1], X[i,j, 1,0], X[i,j, 1,1]
            det = a*d - b*c
            Y[i,j, 0,0] =  d / det
            Y[i,j, 0,1] = -b / det
            Y[i,j, 1,0] = -c / det
            Y[i,j, 1,1] =  a / det
    return Y

#### 3D determinants and inverses

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,:,::1] det_and_inv_3x3(double[:,:,:,:,::1] X, double[:,:,::1] det_out):
    cdef long n0, n1, n2, i0, i1, i2
    cdef double det, invdet
    n0,n1,n2 = X.shape[0], X.shape[1], X.shape[2]
    cdef double x00,x01,x02,x10,x11,x12,x20,x21,x22

    cdef double[:,:,:,:,::1] Y = np.empty_like(X)

    for i0 in prange(n0, nogil=True, schedule='static'):
        for i1 in range(n1):
            for i2 in range(n2):
                x00,x01,x02 = X[i0, i1, i2, 0, 0], X[i0, i1, i2, 0, 1], X[i0, i1, i2, 0, 2]
                x10,x11,x12 = X[i0, i1, i2, 1, 0], X[i0, i1, i2, 1, 1], X[i0, i1, i2, 1, 2]
                x20,x21,x22 = X[i0, i1, i2, 2, 0], X[i0, i1, i2, 2, 1], X[i0, i1, i2, 2, 2]

                det = x00 * (x11 * x22 - x21 * x12) - \
                      x01 * (x10 * x22 - x12 * x20) + \
                      x02 * (x10 * x21 - x11 * x20)

                det_out[i0, i1, i2] = det

                invdet = 1.0 / det

                Y[i0, i1, i2, 0, 0] = (x11 * x22 - x21 * x12) * invdet
                Y[i0, i1, i2, 0, 1] = (x02 * x21 - x01 * x22) * invdet
                Y[i0, i1, i2, 0, 2] = (x01 * x12 - x02 * x11) * invdet
                Y[i0, i1, i2, 1, 0] = (x12 * x20 - x10 * x22) * invdet
                Y[i0, i1, i2, 1, 1] = (x00 * x22 - x02 * x20) * invdet
                Y[i0, i1, i2, 1, 2] = (x10 * x02 - x00 * x12) * invdet
                Y[i0, i1, i2, 2, 0] = (x10 * x21 - x20 * x11) * invdet
                Y[i0, i1, i2, 2, 1] = (x20 * x01 - x00 * x21) * invdet
                Y[i0, i1, i2, 2, 2] = (x00 * x11 - x10 * x01) * invdet

    return Y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:,::1] determinants_3x3(double[:,:,:,:,::1] X):
    cdef size_t n0, n1, n2, i0, i1, i2
    n0,n1,n2 = X.shape[0], X.shape[1], X.shape[2]

    cdef double[:,:,::1] Y = np.empty((n0,n1,n2))
    cdef double[:,::1] x

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                x = X[i0, i1, i2, :, :]

                Y[i0,i1,i2] = x[0, 0] * (x[1, 1] * x[2, 2] - x[2, 1] * x[1, 2]) - \
                              x[0, 1] * (x[1, 0] * x[2, 2] - x[1, 2] * x[2, 0]) + \
                              x[0, 2] * (x[1, 0] * x[2, 1] - x[1, 1] * x[2, 0])
    return Y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,:,::1] inverses_3x3(double[:,:,:,:,::1] X):
    cdef size_t n0, n1, n2, i0, i1, i2
    cdef double det, invdet
    n0,n1,n2 = X.shape[0], X.shape[1], X.shape[2]

    cdef double[:,:,:,:,::1] Y = np.empty_like(X)
    cdef double[:,::1] x, y

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                x = X[i0, i1, i2, :, :]
                y = Y[i0, i1, i2, :, :]

                det = x[0, 0] * (x[1, 1] * x[2, 2] - x[2, 1] * x[1, 2]) - \
                      x[0, 1] * (x[1, 0] * x[2, 2] - x[1, 2] * x[2, 0]) + \
                      x[0, 2] * (x[1, 0] * x[2, 1] - x[1, 1] * x[2, 0])

                invdet = 1.0 / det

                y[0, 0] = (x[1, 1] * x[2, 2] - x[2, 1] * x[1, 2]) * invdet
                y[0, 1] = (x[0, 2] * x[2, 1] - x[0, 1] * x[2, 2]) * invdet
                y[0, 2] = (x[0, 1] * x[1, 2] - x[0, 2] * x[1, 1]) * invdet
                y[1, 0] = (x[1, 2] * x[2, 0] - x[1, 0] * x[2, 2]) * invdet
                y[1, 1] = (x[0, 0] * x[2, 2] - x[0, 2] * x[2, 0]) * invdet
                y[1, 2] = (x[1, 0] * x[0, 2] - x[0, 0] * x[1, 2]) * invdet
                y[2, 0] = (x[1, 0] * x[2, 1] - x[2, 0] * x[1, 1]) * invdet
                y[2, 1] = (x[2, 0] * x[0, 1] - x[0, 0] * x[2, 1]) * invdet
                y[2, 2] = (x[0, 0] * x[1, 1] - x[1, 0] * x[0, 1]) * invdet

    return Y


cpdef double[:,:,:,::1] matmatT_2x2(double[:,:,:,::1] B):
    """Compute B * B^T for each matrix in the input."""
    cdef double[:,:,:,::1] X = np.zeros_like(B, order='C')
    cdef size_t n0 = B.shape[0]
    cdef size_t n1 = B.shape[1]
    for i0 in range(n0):
        for i1 in range(n1):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        X[i0,i1, j,l] += B[i0,i1, j,k] * B[i0,i1, l,k]
    return X

cpdef double[:,:,:,:,::1] matmatT_3x3(double[:,:,:,:,::1] B):
    """Compute B * B^T for each matrix in the input."""
    cdef double[:,:,:,:,::1] X = np.zeros_like(B, order='C')
    cdef size_t n0 = B.shape[0]
    cdef size_t n1 = B.shape[1]
    cdef size_t n2 = B.shape[2]
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            X[i0,i1,i2, j,l] += B[i0,i1,i2, j,k] * B[i0,i1,i2, l,k]
    return X

def matmatT(B):
    if len(B.shape) == 4 and B.shape[-2:] == (2,2):
        return matmatT_2x2(B)
    elif len(B.shape) == 5 and B.shape[-2:] == (3,3):
        return matmatT_3x3(B)
    else:
        assert False, 'matmatT not implemented for shape %s' % B.shape


#### Parallelization

def chunk_tasks(tasks, num_chunks):
    """Generator that splits the list `tasks` into roughly `num_chunks` equally-sized parts."""
    n = len(tasks) // num_chunks + 1
    for i in range(0, len(tasks), n):
        yield tasks[i:i+n]

cdef object _threadpool = None

cdef object get_thread_pool():
    global _threadpool
    if _threadpool is None:
        _threadpool = ThreadPoolExecutor(pyiga.get_max_threads())
    return _threadpool

################################################################################
# Assembler classes (autogenerated)
################################################################################

include "assemblers.pxi"

################################################################################
# Driver routines for assemblers
################################################################################

def generic_vector_asm(kvs, asm, symmetric, format, layout):
    assert layout in ('packed', 'blocked')
    dim = len(kvs)
    bs = tuple(kv.numdofs for kv in kvs)
    bw = tuple(kv.p for kv in kvs)
    mlb = MLBandedMatrix(bs + (dim,), bw + (dim,))
    if dim == 2:
        X = generic_assemble_core_vec_2d(asm, mlb.bidx[:dim], symmetric)
    elif dim == 3:
        X = generic_assemble_core_vec_3d(asm, mlb.bidx[:dim], symmetric)
    else:
        assert False, 'dimension %d not implemented' % dim
    mlb.data = X
    if layout == 'blocked':
        axes = (dim,) + tuple(range(dim))    # bring last axis to the front
        mlb = mlb.reorder(axes)
    if format == 'mlb':
        return mlb
    else:
        return mlb.asmatrix(format)

## 2D

def mass_2d(kvs, geo):
    return generic_assemble_2d_parallel(MassAssembler2D(kvs, geo), symmetric=True)

def stiffness_2d(kvs, geo):
    return generic_assemble_2d_parallel(StiffnessAssembler2D(kvs, geo), symmetric=True)


## 3D

def mass_3d(kvs, geo):
    return generic_assemble_3d_parallel(MassAssembler3D(kvs, geo), symmetric=True)

def stiffness_3d(kvs, geo):
    return generic_assemble_3d_parallel(StiffnessAssembler3D(kvs, geo), symmetric=True)


################################################################################
# Bindings for the C++ low-rank assembler (fastasm.cc)
################################################################################

def fast_mass_2d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef MassAssembler2D asm = MassAssembler2D(kvs, geo)
    return fast_assemble_cy.fast_assemble_2d_wrapper(_entry_func_2d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

def fast_stiffness_2d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef StiffnessAssembler2D asm = StiffnessAssembler2D(kvs, geo)
    return fast_assemble_cy.fast_assemble_2d_wrapper(_entry_func_2d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)


def fast_mass_3d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef MassAssembler3D asm = MassAssembler3D(kvs, geo)
    return fast_assemble_cy.fast_assemble_3d_wrapper(_entry_func_3d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

def fast_stiffness_3d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef StiffnessAssembler3D asm = StiffnessAssembler3D(kvs, geo)
    return fast_assemble_cy.fast_assemble_3d_wrapper(_entry_func_3d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

