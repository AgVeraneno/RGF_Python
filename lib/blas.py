import numba as nb
from numba import cuda as ncuda
import numpy as np
## cupy rewrite ##
import cupy as cp
from cupy.linalg import util
from cupy.core import core
from cupy import cuda
from cupy.cuda import cublas
from cupy.cuda import device
from cupy.linalg import decomposition
from cupy.linalg import util
if cuda.cusolver_enabled:
    from cupy.cuda import cusolver
from skcuda import cusolver as sk_cusolver
    
@ncuda.jit
def inv_cuda(MI, MO):
    TPB = 16
    ## create shared memory array ##
    sM = cuda.shared.array(shape=(TPB, TPB), dtype=nb.complex128)
    ## allocate block ##
    x,y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid
    ## check validity ##
    if x >= MO.shape[0] and y >= MO.shape[1]:
        return
    for i in range(bpg):
        ## Preload data into shared memory ##
        sM[tx, ty] = MI[x, ty + i * TPB]
        cuda.syncthreads()
        
def inv(a):
    """Computes the inverse of a matrix.

    This function computes matrix ``a_inv`` from n-dimensional regular matrix
    ``a`` such that ``dot(a, a_inv) == eye(n)``.

    Args:
        a (cupy.ndarray): The regular matrix

    Returns:
        cupy.ndarray: The inverse of a matrix.

    .. seealso:: :func:`numpy.linalg.inv`
    """
    if not cuda.cusolver_enabled:
        raise RuntimeError('Current cupy only supports cusolver in CUDA 8.0')

    # to prevent `a` to be overwritten
    a = a.copy()

    util._assert_cupy_array(a)
    util._assert_rank2(a)
    util._assert_nd_squareness(a)

    if a.dtype.char == 'f' or a.dtype.char == 'd' or a.dtype.char == 'D':
        dtype = a.dtype.char
    else:
        dtype = np.find_common_type((a.dtype.char, 'f'), ()).char

    cusolver_handle = device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=dtype)

    ipiv = cp.empty((a.shape[0], 1), dtype=dtype)

    if dtype == 'f':
        getrf = cusolver.sgetrf
        getrf_bufferSize = cusolver.sgetrf_bufferSize
        getrs = cusolver.sgetrs
    elif dtype == 'D':
        getrf = sk_cusolver.cusolverDnZgetrf
        getrf_bufferSize = sk_cusolver.cusolverDnZgetrf_bufferSize
        getrs = sk_cusolver.cusolverDnZgetrs
    else:  # dtype == 'd'
        getrf = cusolver.dgetrf
        getrf_bufferSize = cusolver.dgetrf_bufferSize
        getrs = cusolver.dgetrs

    m = a.shape[0]

    buffersize = getrf_bufferSize(cusolver_handle, m, m, a.data.ptr, m)
    workspace = cp.empty(buffersize, dtype=dtype)

    # LU factorization
    getrf(cusolver_handle, m, m, a.data.ptr, m, workspace.data.ptr,
          ipiv.data.ptr, dev_info.data.ptr)

    b = cp.eye(m, dtype=dtype)

    # solve for the inverse
    getrs(cusolver_handle, 0, m, m, a.data.ptr, m, ipiv.data.ptr, b.data.ptr,
          m, dev_info.data.ptr)

    return b
    
if __name__ == '__main__':
    N = 1000
    A = np.ones((N,N), dtype=np.complex128)
    B = cp.ones((N,N), dtype=np.complex128)
    O = cp.eye(N, dtype=cp.float64)
    print(inv(O))
