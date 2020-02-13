import numpy as np
cimport numpy as np
cimport cython
from scipy import sparse, linalg, signal, misc, fftpack, ndimage
import imageio
import numpy as np
cimport numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
 

ctypedef float DTYPE_t
DTYPE = np.float32

cdef add_gaussian_noise(np.ndarray[DTYPE_t, ndim=2] f, float sigma=0.001):
    """
    Adds gaussian noise to image
    """
    cdef shape = [f.shape[0], f.shape[1]]

    cdef np.ndarray[DTYPE_t, ndim=2] out = np.zeros((f.shape[0],f.shape[1]),dtype=DTYPE)
    out = f + sigma * numpy.random.normal(0,1,np.prod(shape)).reshape(shape).astype(DTYPE)
    return out

cdef add_gaussian_blurring(f, ker, sigma):
    out = signal.fftconvolve(f,ker, mode='same')
    return out

cdef np.ndarray[DTYPE_t, ndim=3] grad(np.ndarray[DTYPE_t, ndim=2] f):
    """
    Calculates gradient of image f of size n,m
    returns gradient of size 2,n,m

    """
    cdef int n = f.shape[0] 
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros([2,f.shape[0],f.shape[1]], dtype=DTYPE )

    # x-direction
    out[0, :n-1, :] = f[1:, :] - f[:n-1, :]

    # y-direction
    out[1, :, :n-1] = f[:, 1:] - f[:, :n-1]
    return out

cdef np.ndarray[DTYPE_t, ndim=2] div(np.ndarray[DTYPE_t, ndim=3] f):
    """
    Calculates divergence of image f of size 2,n,m
    returns divergence of size n,m
    """
    cdef int n = f.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros((2, n, n), dtype=DTYPE)

    # Boundaries along y-axis
    out[0, 0, :] = f[0, 0, :]
    out[0, n-1, :] = -f[0, n-2, :]

    # Inside along y-axis
    out[0, 1:n-1, :] = f[0, 1:n-1, :] - f[0, :n-2, :]

    # Boundaries along y-axis
    out[1, :, 0] = f[1, :, 0]
    out[1, :, n-1] = -f[1, :, n-2]
    # Inside along y-axis
    out[1, :, 1:n-1] = f[1, :, 1:n-1] - f[1, :, :n-2]

    # Return sum along x-axis
    return np.sum(out, axis=0)

cdef norm1(f, axis=0, keepdims=False):
    """
    returns 1-norm of image f of size n,m
    returns number
    """
    return np.sqrt(np.sum(f**2, axis=axis, keepdims=keepdims))

cdef better_norm1(f):
    return np.sqrt(f[0,:,:]**2 + f[1,:,:]**2)

def ChambollePock_denoise(np.ndarray[DTYPE_t, ndim=2] f, float lam, float tau = 0.50, float sig = 0.30, float theta = 1.0, bint acc = False, float tol = 1.0e-5):
    cdef np.ndarray[DTYPE_t, ndim=3] p = np.zeros((2, f.shape[0], f.shape[1]), DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] p_hat = np.zeros((2, f.shape[0], f.shape[1]), DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] divp = np.zeros((f.shape[0], f.shape[1]), DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] u = np.zeros((f.shape[0], f.shape[1]), DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] u_prev = np.zeros((f.shape[0], f.shape[1]), DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] u_hat = np.zeros((f.shape[0], f.shape[1]), DTYPE)

    cdef int maxiter = 100
    #costlist = np.zeros(maxiter+1)
    #costlist[0] = cost(u,f,lam)
    cdef float r = np.linalg.norm((u-f) - lam*div(p))
    cdef float gam
    if acc:
        gam = 1.0*lam
    cdef np.ndarray[DTYPE_t, ndim=2] f_over_lam = f/lam
    cdef int i
    divp = div(p) 
    for i in range(maxiter):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = p_hat / np.maximum(1.0, better_norm1(p_hat))
        divp = div(p)
        u = lam*tau/(lam + tau) * (u/tau + f_over_lam + divp) 
        print(i, np.linalg.norm((u - f) - lam*divp)/r)
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
        #costlist[i+1] = cost(u,f,lam)
    return u

def main():
    cdef np.ndarray[DTYPE_t, ndim = 2] Lenna
    Lenna  = imageio.imread('images/lenna.ppm', pilmode = 'F')/255.0
    #lines = np.random.randint(0,512, 200)
    #Lenna[lines,:] = 0.0
    #Lenna = ndimage.gaussian_filter(Lenna, 5) 
    Lenna = add_gaussian_noise(Lenna,0.1)
    #lambdas = [0.1, 0.2, 0.5, 1.0]
    #vmin = np.min(Lenna); vmax = np.max(Lenna)
    #plt.imshow(Lenna, cmap = 'gray')
    #plt.show()

    lam = 0.1
    t = time.time()
    result = ChambollePock_denoise(Lenna, lam, acc=True)
    print(time.time() - t)
    plt.imshow(result, cmap = 'gray')
    plt.show()

    #for i,lam in enumerate(lambdas):
    #    result, _ = ChambollePock_denoise(Lenna, lam) 
    #    #result = ptv.tv1_2d(Lenna, lam)
    #    plt.imsave('results/Lenna_' + str(i) + '.png', result, cmap = 'gray')#, vmin = vmin, vmax = vmax)
    return 0
main()
