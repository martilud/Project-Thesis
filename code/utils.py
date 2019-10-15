import os
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from termcolor import colored
import time
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
from scipy.linalg import circulant, toeplitz
import scipy.misc

def unit_vector(vector):
    """ 
    Normalises a vector  
    """
    return vector / np.linalg.norm(vector)

def project_them(Q, X):
    """ 
    Projecting to a subspace given by orthonormal rows of a matrix Q 
    """
    P = np.dot(Q.T, Q)
    return np.dot(P, X.T).T

def safe_norm_div(x, y):
    """ 
    Calculates ||x-y||/||y|| safely 
    """
    if (np.linalg.norm(y) == 0.0):
        return 0.0
    return np.linalg.norm(x - y) / np.linalg.norm(y)


def rank_r(m, d, r):
    """
    create a unit norm matrix of a certain rank
    """
    U, ss, V = np.linalg.svd(np.random.randn(m, d))
    idx = np.sort(np.random.choice(min(m, d), r, replace = False))
    S = np.zeros((m, d))
    S[idx, idx] = ss[idx]
    A = np.dot(U, np.dot(S, V))
    A /= np.linalg.norm(A, 2)
    return A

def set_it_up(m, d, rank = False, h = 5, N_train = 50, N_test = 1, sigma = 0.05):
    """
    Create the matrix A of size m x d, and a certain rank
    Create N_train training data X_i^train, Y_i^train, with noise level sigma
    Create N_test testing data X_i^test, Y_i^test
    """

    # Create the matrix A
    if rank == False:
        A = np.random.randn(m ,d)
        A /= np.linalg.norm(A, 2)
    else:
        A = rank_r(m, d, rank)

    # Create the training data
    mean_X = np.zeros(d)
    cov_X = np.zeros((d, d))
    cov_X[range(h), range(h)] = 1.0 # set the data dimensionality
    X = np.random.multivariate_normal(mean_X, cov_X, N_train)

    # Noise
    mean_eta = np.zeros(m)
    cov_eta = np.eye(m)
    Eta = sigma * np.random.multivariate_normal(mean_eta, cov_eta, N_train)

    # Create Y_i = A X_i + Eta_i
    Y = np.zeros((N_train, m))
    for i in range(N_train):
        Y[i, :] = np.dot(A, X[i, :]) + Eta[i, :]

    ## Create the testing data
    X_test = np.random.multivariate_normal(mean_X, cov_X, N_test)
    Eta_test = sigma * np.random.multivariate_normal(mean_eta, cov_eta, N_test)
    Y_test = np.zeros((N_test, m))
    for i in range(N_test):
        Y_test[i, :] = np.dot(A, X_test[i, :]) + Eta_test[i, :]

    ## Create empirical projection from the training data
    piY_n = pi_hat_n(covariance(Y), h = h)

    return A, piY_n, X_test, Y_test

def set_it_up_id(n,  h = 5, N_train = 50, N_test = 1, sigma = 0.05):
    """
    MARTIN
    Same as set_it_up but uses A = id
    Also "spreads" the nonzero elements of the covariance matrix
    """
    # Create the training data
    mean_X = np.zeros(n)
    ones = np.zeros(n)
    ones[range(h)] = 1.0
    diag = np.random.choice(ones,n,False)
    cov_X = np.diag(diag) # set the data dimensionality
    X_train = numpy.random.multivariate_normal(mean_X, cov_X, N_train)

    # Noise
    mean_eta = 0
    cov_eta = 1
    Eta_train = sigma * numpy.random.normal(mean_eta, cov_eta, N_train)

    # Create Y_i = X_i + Eta_i
    Y_train = np.zeros((N_train, n))
    Y_train = X_train + Eta_train

    # Create the testing data
    X_test = np.random.multivariate_normal(mean_X, cov_X, N_test)
    Eta_test = sigma * np.random.normal(mean_eta, cov_eta, N_test)
    Y_test = np.zeros((N_test, n))
    Y_test = X_test + Eta_test

    # Create empirical projection from the training data
    piY_n = pi_hat_n(covariance(Y_train), h = h)

    return piY_n, X_test, Y_test

def set_it_up_image(images, amount, shape, h, N_train, N_test, sigma = 0.1):
    N_image = len(images)
    n = 256
    test = np.array((n**2, n**2))
    print("WORKED")
    image_array = np.empty((N_image,) + shape)
    X_train = np.empty((N_train,) + shape)
    X_test = np.empty((N_test,) + shape)
    Y_train = np.empty((N_train,np.prod(shape)))
    Y_test = np.empty((N_test,) + shape)
    for i in range(N_image):
        image_array[i] = scipy.misc.imread('images/' + images[i], 'gray')/255.0

    for i in range(N_image):
        indice = 0
        for j in range(amount[i]):
            X_train[indice,:,:] = image_array[i]
        indice += amount[i]
    for i in range(N_train):
        Y_train[i,:] = X_train[i,:,:].reshape(np.prod(shape)) + sigma * np.random.normal(0,1,np.prod(shape))
    X_test[0,:,:] = image_array[0]
    Y_test[0,:,:] = X_test[0,:,:] + sigma * np.random.normal(0,1,np.prod(shape)).reshape(shape) 
    print(Y_train.shape)
    piY_n = pi_hat_n(covariance(Y_train[:,:(500*500)]), h = shape[0])

    return piY_n, X_test, Y_test

def empirical_estimators(A, Y, Pi_n):
    """
    for a given A, Y compute the empirical estimator

    """
    N = Y.shape[0]
    X = np.zeros((N, A.shape[1]))
    Eta = np.zeros((N, A.shape[0]))
    Ainv = np.linalg.pinv(A)

    for i in range(N):
        Pi_n_Y = np.dot(Pi_n, Y[i, :])
        X[i, :] = np.dot(Ainv, Pi_n_Y)
        Eta[i, :] = Y[i, :] - Pi_n_Y

    return X, Eta

def soft_thresholding(lam, vec):
    """
    soft thresholding a vector with a threshold being lam
    TODO: replace with inbuilt func
    """
    s_tv = np.zeros(vec.shape)
    # import pdb; pdb.set_trace()
    for i in range(len(vec)):
        if vec[i] > lam / 2.0:
            s_tv[i] = vec[i] - lam / 2.0
        elif vec[i] < - lam / 2.0:
            s_tv[i] = vec[i] + lam / 2.0
    return s_tv

def sgn(v):
    """
    return sign of a vector
    """
    return np.sign(v)

def positive_part(v):
    """
    returns the positive part of a function
    """
    u = np.zeros(len(v))
    return np.maximum(v, u)

def fixed_point_iterations_lam(A, y, lam, alpha, z = None, max_iter = 500):
    """
    fixed point iterations for the elastic net solution
    """
    if z is None:
        z = np.zeros((A.shape[1], 1))

    u, s, v = np.linalg.svd(np.dot(A.T, A))
    tau = (np.sum(s) + s[-1]) / 2.0
    M = np.dot(A.T, A)
    Ay = np.dot(A.T, y)
    Mat = tau * np.eye(M.shape[0]) - M
    factor = 1.0 / (tau + alpha * lam)
    error = np.zeros(max_iter - 1)

    for i in range(max_iter):
        old_z = np.copy(z)
        # import pdb; pdb.set_trace()
        z = factor * soft_thresholding(lam, np.dot(Mat, z) + Ay)
    return z

def covariance(Z):
    """
    straightforward computation of the empirical covariance
    """
    n = Z.shape[0]
    S = np.zeros((Z.shape[1], Z.shape[1]))
    for i in range(n):
        np.outer(Z[i, :], Z[i, :]) 
        #S = np.outer(Z[i, :], Z[i, :])
        S += 1.0 / n * S
    return S

def pi_hat_n(S, h = 1):
    """
    compute the empirical projection of a given rank h
    """
    eigvals, eigvectors = np.linalg.eigh(S)
    eigvectors = eigvectors[:, -h:].T[::-1].T
    P = eigvectors.dot(eigvectors.T)
    return P


def elastic_norm(x, alpha = 1):
    """
    compute |x|_1 + alpha |x|_2^2
    """
    return np.linalg.norm(x, 1) + alpha * np.linalg.norm(x, 2)

def add_gaussian_noise(f, sigma=0.001):
    """
    Adds gaussian noise to image
    """
    out = np.zeros((2,) + f.shape, f.dtype)

    shape = f.shape

    out = f + sigma* numpy.random.normal(0,1,shape[0]*shape[1]).reshape(shape)
    return out

def add_gaussian_blurring(f, ker, sigma):
    out = signal.fftconvolve(f,ker, mode='same')
    return out

def grad(f):
    """
    Calculates gradient of image f of size n,m
    returns gradient of size 2,n,m

    """
    out = np.zeros((2,) + f.shape, f.dtype)

    # x-direction
    out[0, :-1, :] = f[1:, :] - f[:-1, :]

    # y-direction
    out[1, :, :-1] = f[:, 1:] - f[:, :-1]
    return out

def better_grad(f):
    out = np.zeros((2,) + f.shape, f.dtype)

    # x-direction
    out[0,0, :] = f[1,:] - f[0, :]
    out[0,-1, :] = f[-1,:] - f[-2, :]
    out[0, 1:-1, :] = (f[2:, :] - f[:-2, :])/2

    # y-direction
    out[0,:, 0] = f[:, 1] - f[:, 0]
    out[0,:, -1] = f[:,-1] - f[:, -2]

    out[1, :, 1:-1] = (f[:, 2:] - f[:, :-2])/2
    return out

def div(f):
    """
    Calculates divergence of image f of size 2,n,m
    returns divergence of size n,m
    """
    out = np.zeros_like(f)

    # Boundaries along y-axis
    out[0, 0, :] = f[0, 0, :]
    out[0, -1, :] = -f[0, -2, :]
    # Inside along y-axis
    out[0, 1:-1, :] = f[0, 1:-1, :] - f[0, :-2, :]

    # Boundaries along y-axis
    out[1, :, 0] = f[1, :, 0]
    out[1, :, -1] = -f[1, :, -2]
    # Inside along y-axis
    out[1, :, 1:-1] = f[1, :, 1:-1] - f[1, :, :-2]

    # Return sum along x-axis
    return np.sum(out, axis=0)

def norm1(f, axis=0, keepdims=False):
    """
    returns 1-norm of image f of size n,m
    returns number
    """
    return np.sqrt(np.sum(f**2, axis=axis, keepdims=keepdims))

def better_norm1(f):
    return np.sqrt(f[0,:,:]**2 + f[1,:,:]**2)

def TV(u):
    return np.sum(better_norm1(grad(u)))

def anisotropic_TV(u):
    return np.sum(np.abs(u[1:,:] - u[:-1,:])) + np.sum(np.abs(u[:,1:] - u[:,:-1]))

def cost(u,f,lam):
    return 0.5* np.linalg.norm(u-f) + lam*TV(u) 

def cost_matrix(A,u,f,lam):
    return 0.5* np.linalg.norm(A.dot(u.reshape(f.shape[0]*f.shape[1])).reshape(f.shape)-f) + lam*TV(u) 

def psnr(noisy,true):
    """
    Calculates psnr between two images
    """
    mse = np.mean((noisy-true)**2)
    return 20*np.log10(mse) - 10*np.log10(np.max(np.max(noisy),np.max(true)))

