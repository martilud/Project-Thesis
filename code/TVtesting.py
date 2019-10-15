import os
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import time
from matplotlib import rc
import matplotlib as mpl
from scipy.stats import ortho_group

from utils import *
from ChambollePock import *

def R(t, Y, X, Z = None):
    if Z is None:
        Z, _ = ChambollePock_denoise(Y,(1-t)/t)
    return np.linalg.norm(Z - X)**2

def gridSearch(Y_test,X,plot=True):
    N = 100
    t_list = np.linspace(0.0,1.0, N)
    R_listtrue = np.zeros(N)
    #R_listhat = np.zeros(N)
    for i in range(1,N-1):
        print(i)
        R_listtrue[i] = R(t_list[i],Y_test,X)
        #R_listhat[i] = R2D(t_list[i],Y_test,X_hat)
    t_opttrue = t_list[np.argmin(R_listtrue[1:N-1])]
    #t_opthat = t_list[np.argmin(R_listhat[1:N-1])]
    if plot:
        plt.plot(t_list[1:N-1],R_listtrue[1:N-1],label="True", color= "red")
        plt.plot(t_opttrue,np.min(R_listtrue[1:N-1]), 'ro')
        #plt.plot(t_list[1:N-1],R_listhat[1:N-1],label="Empirical", color = "blue")
        #plt.plot(t_opthat,np.min(R_listhat[1:N-1]), 'bo')
        plt.xlabel('t')
        plt.ylabel('R(t)')
        plt.legend()
        plt.show()
    print("optimal lambda", (1 - t_opttrue)/t_opttrue)
    return t_opttrue



n = 100
nn = n*n
h = 50
N_train = 10
N_test = 1
sigma = 0.1

images = ['lenna.ppm','bridge.pgm']
amount = [5,5]
set_it_up_image(images, amount, (512,512), 512, N_train, N_test)
#X_test = misc.imread('images/Lenna.jpg', 'jpg')
X_test= X_test/255.0

#X_test = np.zeros((n,n))
#X_test[24:75,24:75] = 1.0
sigma = 0.2
Y_test = add_gaussian_noise(X_test,sigma)
#Pi, X_test, Y_test = set_it_up_id(nn,h,N_train,N_test,sigma)
#X_hat = np.dot(Y_test, Pi)
##X_test = X_test.reshape((n,n))#X_test.reshape((N_test,n,n))
##Y_test = Y_test.reshape((n,n))#Y_test.reshape((N_test,n,n))
##X_hat = X_hat.reshape((n,n))
#plt.plot(Y_test.T)
#plt.plot(X_hat.T)
#plt.show()

#fig = plt.figure()
#ax = fig.subplots(1,3)
#ax[0].imshow(X_test, cmap = 'magma')
#ax[0].set_title("X_test")
#ax[1].imshow(Y_test, cmap ='magma')
#ax[1].set_title("Y_test")
#ax[2].imshow(X_hat, cmap = 'magma')
#ax[2].set_title("X_hat")
#plt.show()

t = time.time()
t_opt = gridSearch(Y_test,X_test, True)
print(time.time() - t)
t = time.time()

result, _ = ChambollePock_denoise(Y_test, (1-t_opt)/t_opt)

fig = plt.figure()
ax = fig.subplots(1,3)
ax[0].imshow(X_test, cmap = 'gray')
ax[0].set_title("X_test")
ax[1].imshow(Y_test, cmap ='gray')
ax[1].set_title("Y_test")
ax[2].imshow(result, cmap = 'gray')
ax[2].set_title("Result")
plt.show()

#t_optalg = backtracking_interpolate(Y_test, X_hat, tol = 1e-6)

#print(time.time() - t)
#methods = ['discrepancy_ruleTV', 'monotone_error_ruleTV']
#for method in methods:
#    t = time.time()
#    z, lam = eval(method)(Y_test,sigma)
#    print(method, "lambda: ", lam, "time: ", time.time()-t)
#
#plt.plot(X_hat, label="X HATTE")
#plt.plot(X_test.T, label="X TRU")
#plt.legend()
#plt.show()


#plt.plot(result, label='result')
#plt.plot(X_train[0,:], label= 'not result')
#plt.legend()
#plt.show()
