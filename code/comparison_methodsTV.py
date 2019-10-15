import os
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import time
from matplotlib import rc
import matplotlib as mpl
from scipy.stats import ortho_group

from utils import *
import prox_tv as ptv

def discrepancy_ruleTV(Y, noise_std, tau = 1.0, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the discrepancy principle
	noise_std is the estimate (or the true value) of the noise parameter
	"""

	m = Y.shape[0]
        A = np.identity(m)
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = tau * noise_std * np.sqrt(m)
	tol = 1e-5
	while True:
		z = ptv.tv1_2d(Y, current_lam)
		if np.linalg.norm(np.dot(A,z) - Y) < error_level:
			lam_final = current_lam
			break
		current_lam = lam_not * (q ** iter)
		if (current_lam <= tol) or (iter > max_iter):
			lam_final = current_lam
			break
		iter += 1
	return z, lam_final

def monotone_error_ruleTV(Y, noise_std, tau = 1.0, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the monotone error rule
	"""
	m = Y.shape[0]
        A = np.identity(m)
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = tau * noise_std * np.sqrt(m)
	tol = 1e-5

	while True:
		z_new = ptv.tv1_2d(Y, current_lam)
		iter += 1

		if iter == 1:
			z_old = z_new
			current_lam = lam_not * (q ** iter)
			continue

		r_temp, _, _, _ = np.linalg.lstsq(A.T, z_old - z_new,rcond=False)
		if np.linalg.norm(r_temp) == 0.0:
			error_arg = np.linalg.norm(np.dot(A, z_new) - Y)
		else:
			error_arg = np.abs(np.dot(np.dot(A, z_new) - Y, r_temp.T) / np.linalg.norm(r_temp))
		if (error_arg < error_level):
			lam_final = current_lam
			break
		current_lam = lam_not * (q ** iter)
		if (current_lam <= tol) or (iter > max_iter):
			lam_final = current_lam
			break

	return z_new, lam_final

def backtracking_interpolate(Y, estimator, tol = 1e-6):
	"""
	Implementation of backtracking line search with interpolation
	"""

	h = - 0.001
	satisfied = False
	cnt = -1
	max_iter = 100
	beta = 0.8
	go = True
	alpha0 = 0.01
	tau = 0.45
	c1 = 1e-2
	t = np.zeros(max_iter + 1); t[0] = 1.0
	t_current = t[0]

	descent_directions = np.zeros(len(t))
	loss_functional_evals = np.zeros(len(t))
	alphas = np.zeros(len(t))

	while satisfied == False:
		# Compute gradient at t
		cnt += 1
		z = ptv.tv1_1d(Y,(1 - t_current)/t_current)
                t_direction = t_current + h
		z_h =ptv.tv1_1d(Y,(1 - t_direction)/t_direction)
                descent_directions[cnt] =  (np.linalg.norm(z - estimator) - np.linalg.norm(z_h - estimator)) / h
		loss_functional_evals[cnt] =  np.linalg.norm(z - estimator)

		# Check stopping criteria
		if (np.abs(descent_directions[cnt]) < tol) or (cnt >= max_iter):
			satisfied = True
			break

		# Find the step-size
		alpha = np.copy(alpha0)
		tau = - np.abs(descent_directions[cnt]) ** 2

		if cnt == 0 :
			last_alpha = alpha
		t_temp = t_current + alpha * descent_directions[cnt]
		phi0 = loss_functional_evals[cnt]
		phiprime0 = tau
		phialpha = np.linalg.norm(ptv.tv1_1d(Y, (1 - t_temp) / t_temp) - estimator)

		if (phialpha - phi0) < (c1 * phiprime0 * alpha):
			step_size = alpha
			# print "here and {0}, {1}, {2}, with finally {3}".format(phialpha - phi0, phi0, alpha, c1 * phiprime0 * alpha)
		else:
			step_size = - (alpha ** 2 * phiprime0) / (2.0 * (phialpha - phi0 - phiprime0 * alpha))
			# print "there {0}".format(step_size)
		if (np.abs(last_alpha / step_size) > 10.0) or (np.abs(step_size) < 1e-3):
			# print "{0} and {1}".format(np.abs(step_size - last_alpha), np.abs(step_size / last_alpha))
			step_size = last_alpha * beta * 1.3

		last_alpha = np.copy(step_size)
		t_new = t_current + step_size * descent_directions[cnt]
		t[cnt + 1] = t_new
		t_current = t_new
	#print "cnt is ", cnt
	return t_new

