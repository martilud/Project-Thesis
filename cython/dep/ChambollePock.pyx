from utils import *
#from projectedGradient import *

#def A(f,ker):
#    return fftpack.idctn(ff*fker, shape=s, norm='ortho') #signal.fftconvolve(f,ker,mode='same')

def ChambollePock_denoise(np.ndarray[DTYPE_t, ndim=2] f, float lam, float tau = 0.50, float sig = 0.30, float theta = 1.0, bool acc = False, float tol = 1.0e-5):
    cdef np.ndarray[DTYPE_t, ndim=3] p = np.zeros((2, f.shape[0], f.shape[1], DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] p_hat = np.zeros((2, f.shape[0], f.shape[1], DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] u = np.zeros(f.shape[0], f.shape[1], DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] u_prev = np.zeros(f.shape[0], f.shape[1], DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] u_hat = np.zeros(f.shape[0], f.shape[1], DTYPE)
    int maxiter = 100
    #costlist = np.zeros(maxiter+1)
    #costlist[0] = cost(u,f,lam)
    float r = np.linalg.norm((u-f) - lam*div(p))
    if acc:
        float gam = 1.0*lam
    cdef np.ndarray[DTYPE_t, ndim=2] f_over_lam = f/lam
    int i
    for i in range(maxiter):
        u_prev = np.copy(u)
        p_hat = p + sig*grad(u_hat)
        p = p_hat / np.maximum(1.0, better_norm1(p_hat))
        u = lam*tau/(lam + tau) * (u/tau + f_over_lam + div(p)) 
        print(i, np.linalg.norm((u - f) - lam*div(p))/r)
        if (acc):
           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
           tau = theta*tau
           sig = sig/theta
        u_hat = u + theta*(u - u_prev)
        #costlist[i+1] = cost(u,f,lam)
    return u

#def ChambollePock_matrix(f, lam, A, tau = 0.10, sig = 0.30, theta = 1.0, acc = False, tol = 1.0e-7):
#    shape = f.shape
#    dtype = f.dtype
#    p = np.zeros((2,) + shape, dtype)
#    p_hat = np.zeros((2,) + shape, dtype)
#    u = np.copy(f)
#    u_prev = np.copy(f)
#    u_hat = np.copy(f)
#
#    mat = scipy.sparse.eye(shape[0]*shape[1]) + tau/lam * A.dot(A.H)     
#
#    maxiter = 100
#    costlist = np.zeros(maxiter+1)
#    costlist[0] = cost_matrix(A,u,f,lam)
#    if acc:
#        gam = 0.3*lam
#    rhs_f = tau/lam*(A.H).dot(f.reshape(shape[0]*shape[1]))
#    for i in range(maxiter):
#        u_prev = np.copy(u)
#        p_hat = (p + sig*grad(u_hat))
#        p = p_hat / np.maximum(1.0, better_norm1(p_hat))
#        u, _ = scipy.sparse.linalg.cg(mat, (u + tau*div(p)).reshape(shape[0]*shape[1])+rhs_f, x0 = u.reshape(shape[0]*shape[1]), tol = tol)
#        u = u.reshape(shape)
#        if (acc):
#           theta = 1.0/np.sqrt(1.0 + 2.0*gam*tau) 
#           tau = theta*tau
#           sig = sig/theta
#        u_hat = u + theta*(u - u_prev)
#        costlist[i+1] = cost_matrix(A,u,f,lam)
#    return u, costlist

#def ChambollePock_conv(f, lam, tau = 0.1, sig = 0.1, theta = 1.0, tol = 1.0e-5):
#    p = np.zeros((2,) + f.shape, f.dtype)
#    p_hat = np.zeros((2,) + f.shape, f.dtype)
#    u = np.zeros(f.shape, f.dtype)
#    u_prev = np.zeros(f.shape, f.dtype)
#    u_bar = np.zeros(f.shape, f.dtype)
#    u_hat = np.zeros(f.shape, f.dtype)
#    maxiter = 10
#    for i in range(maxiter):
#        u_prev = u
#        #p = np.maximum(1, np.minimum(-1, p + sig*lam*grad(u_hat)))
#        p_hat = p + sig*lam*grad(u_hat)
#        p = p_hat/np.max(1,better_norm1(p_hat))
#        u_bar = u - tau*lam*div(p)
#        print(u_bar.shape)
#        u = fftpack.idctn(np.divide(tau*ff*fker_conj + fftpack.dctn(u_bar,shape=s, norm='ortho'),(tau*fker2 + 1)),shape=s, norm='ortho' ) 
#        print(u)
#        plt.imshow(u,cmap='gray')
#        plt.show()
#        u_hat = u + theta*(u - u_prev)
#    return u

#def createLaplacian(N):
#    diag=np.ones([N*N])
#    mat=scipy.sparse.spdiags([diag,-2*diag,diag],[-1,0,1],N,N)
#    I=scipy.sparse.eye(N)
#    return scipy.sparse.kron(I,mat,format='csr')+scipy.sparse.kron(mat,I,format='csr')
if __name__ == "__main__":
    Originalf = imageio.imread('images/Lenna.jpg', pilmode = 'F')
    Originalf = Originalf/255.0
    f = np.copy(Originalf)
    #f = np.zeros((100,100))
    #f[25:74,25:75] = 1.0
    shape = f.shape
    nx = f.shape[0]
    ny = f.shape[1]
    #A = createLaplacian(nx)
    A = scipy.sparse.eye(f.shape[0]**2)
    sigma = 0.1
    f = add_gaussian_noise(f,sigma)
    lam = 0.5
    result,costlist = ChambollePock_matrix(f, lam, A, tau = 0.10, sig = 0.10, theta = 1.0)
    result1,costlist1 = ChambollePock_denoise(f,lam, tau=0.1, sig = 0.1, theta = 1.0)
    plt.semilogy(costlist, '-')
    plt.semilogy(costlist1)
    plt.show()
    fig = plt.figure()
    ax = fig.subplots(1,2)
    pos = ax[0].imshow(result, cmap = 'gray')
    pos = ax[1].imshow(result1, cmap ='gray')
    fig.colorbar(pos)
    plt.show()
    #hx = 1./(nx+1)
    #hy = 1./(ny+1)
    #dim = 2
    #sigma = 0.1
    #kx = 10
    #ky = 10
    #f = add_gaussian_noise(f,sigma)
    ##ker = np.outer(signal.gaussian(2*nx,sigma), signal.gaussian(2*ny, sigma))
    ##ker = ker[nx:, ny:]
    ##ker = ker/np.sum(ker)
    ##print(ker)
    ##plt.imshow(ker,cmap='gray')
    ##plt.show()
    ##s = [n + k - 1 for n,k in zip(f.shape, ker.shape)]
    ##s = np.array(f.shape) #+ np.array(ker.shape) - 1
    ##
    ##fker = fftpack.dctn(ker,shape=s, norm='ortho')
    ##fker2 = np.abs(fker)**2
    ##
    ##fker_conj = fftpack.dctn(ker[::-1],shape=s, norm='ortho')
    ##
    ##ff = fftpack.dctn(f, shape=s, norm='ortho')
    ##f = A(f,ker)
    ##print(f)
    ##plt.imshow(f,cmap='gray')
    ##plt.show()
    #
    #lam = 0.1
    #time1 = time.time()
    #result,costlist = ChambollePock_denoise(f, lam)
    #time1 = time.time() - time1
    #time2 = time.time()
    #result_acc,costlist_acc = ChambollePock_denoise(f, lam,tau = 0.1, acc = True)
    #time2 = time.time() - time2
    ##time3 = time.time()
    ##result_other,costlist_other = projected_gradient(f, lam)
    ##time3 = time.time() - time3
    #
    #plt.plot(costlist)
    #plt.plot(costlist_acc)
    ##plt.plot(costlist_other)
    #plt.show()
    #print(cost(result,f,lam), np.linalg.norm(result - Originalf), time1)
    #print(cost(result_acc,f,lam),np.linalg.norm(result_acc - Originalf), time2)
    ##print(cost(result_other,f,lam),np.linalg.norm(result_other - Originalf), time3)
    #fig = plt.figure()
    #ax = fig.subplots(1,3)
    #ax[0].imshow(f, cmap = 'gray')
    ##ax[1].imshow(result_other, cmap = 'gray')
    #ax[1].imshow(result, cmap = 'gray')
    #ax[2].imshow(result_acc, cmap = 'gray')
    #plt.show()

