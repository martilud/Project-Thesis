import os
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import time
from matplotlib import rc
import matplotlib as mpl
from scipy.stats import ortho_group

from utils import *

def estimate_rhon(A, Y_train, current_lam, alpha):
	n_train, m, d = Y_train.shape[0], A.shape[0], A.shape[1]
	z = np.zeros((n_train, d))
	error = 0.0
	for iter in range(n_train):
		z[iter, :] = fixed_point_iterations_lam(A, Y_train[iter], lam = current_lam, alpha = alpha)
		for jter in range(iter):
			error += np.linalg.norm(z[iter, :] - z[jter, :]) ** 2
	if n_train > 2:
		n_comb = (n_train - 1) * n_train
	elif n_train == 2:
		n_comb = 1
	else:
		print "ERROR!"
	return error / np.float(n_comb)

def white_rhon_sq(A, current_lam, alpha, n_train = 4):
	m = A.shape[0]
	rhon = 0.0

	for iter in range(n_train):
		xi_temp = np.random.normal(size = m)
		z_temp = fixed_point_iterations_lam(A, xi_temp, lam = current_lam, alpha = alpha)
		rhon += np.linalg.norm(z_temp) ** 2
	rhon /= np.float(n_train)
	return rhon

def discrepancy_rule(A, Y, alpha, noise_std, tau = 1.5, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the discrepancy principle
	noise_std is the estimate (or the true value) of the noise parameter
	"""

	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = tau * noise_std * np.sqrt(m)
	tol = 1e-5
	while True:
		z = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		if np.linalg.norm(np.dot(A, z) - Y) < error_level:
			lam_final = current_lam
			break
		current_lam = lam_not * (q ** iter)
		if (current_lam <= tol) or (iter > max_iter):
			lam_final = current_lam
			break
		iter += 1
	return z, lam_final

def monotone_error_rule(A, Y, alpha, noise_std, tau = 1.5, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the monotone error rule
	"""
	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = tau * noise_std * np.sqrt(m)
	tol = 1e-5

	while True:
		z_new = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		iter += 1

		if iter == 1:
			z_old = z_new
			current_lam = lam_not * (q ** iter)
			continue

		r_temp, _, _, _ = np.linalg.lstsq(A.T, z_old - z_new)
		if np.linalg.norm(r_temp) == 0.0:
			error_arg = np.linalg.norm(np.dot(A, z_new) - Y)
		else:
			error_arg = np.abs(np.dot(np.dot(A, z_new) - Y, r_temp) / np.linalg.norm(r_temp))

		if (error_arg < error_level):
			lam_final = current_lam
			break
		current_lam = lam_not * (q ** iter)
		if (current_lam <= tol) or (iter > max_iter):
			lam_final = current_lam
			break

	return z_new, lam_final

def alternative_monotone_error_rule(A, Y, alpha, noise_std, tau = 1.5, lam_not = 2.0, q = 0.9):
	"""
	Implementation of a different version of the monotone error rule
	"""

	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = tau * noise_std * np.sqrt(m)
	tol = 1e-5

	while True:
		z_new = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		iter += 1
		if iter == 1:
			z_old = z_new
			current_lam = lam_not * (q ** iter)
			continue

		r_temp, _, _, _ = np.linalg.lstsq(A.T, z_old - z_new)
		if np.linalg.norm(r_temp) == 0.0:
			error_arg = np.linalg.norm(np.dot(A, z_new + z_old) * 0.5 - Y)
		else:
			error_arg = np.abs(np.dot(np.dot(A, z_new + z_old) * 0.5 - Y, r_temp) / np.linalg.norm(r_temp))

		if error_arg  < error_level:
			lam_final = current_lam
			break

		current_lam = lam_not * (q ** iter)
		if (current_lam <= tol) or (iter > max_iter):
			lam_final = current_lam
			break

	return z_new, lam_final


def quasi_optimality(A, Y, alpha, lam_not = 2.0, q = 0.9):
	"""
	Implementation of quasi optimality for parameter selection
	"""

	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100


	error = np.zeros(max_iter)
	alt_error = np.zeros(max_iter)
	z_old = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)

	for i in range(1, max_iter):
		current_lam = lam_not * (q ** i)
		z_new = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		error[i]  = np.linalg.norm(z_old - z_new)
		alt_error[i] = np.linalg.norm(z_old - z_new) / current_lam
		z_old = np.copy(z_new)

	opt_idx = np.argmin(error[error != 0.0])
	lam_final = lam_not * (q ** opt_idx)
	z_final = fixed_point_iterations_lam(A, Y, lam = lam_final, alpha = alpha)

	return z_final, lam_final


def lorenzo_balancing(A, Y, alpha, lam_not = 2.0, q = 0.9, good_iter = 41):
	"""
	Implementation of balancing principle for elastic nets, as descrbied in a
	paper by lorenzo rosasco, ernesto de vito, et al
	"""
	m = A.shape[1]
	iter = 0
	old_lam = lam_not * (q ** iter)
	max_iter = 100

	important_constant = 4.0 / 10000.0

	z_old = fixed_point_iterations_lam(A, Y, lam = old_lam, alpha = alpha)
	error = np.zeros(max_iter)
	conditions = np.zeros(max_iter)
	conditions[iter] = (4.0 * important_constant) / (np.sqrt(m) * old_lam * alpha)
	while True:
		iter += 1
		current_lam = lam_not * (q ** iter)
		z_new = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		temp_error  = np.linalg.norm(z_old - z_new)
		condition = (4.0 * important_constant)/ (np.sqrt(m) * old_lam * alpha)

		error[iter]  = temp_error
		conditions[iter] = condition

		if iter < 3:
			z_old = np.copy(z_new)
			old_lam = np.copy(current_lam)
			continue
		if temp_error > condition:
			lam_final = current_lam
			break
		if iter >= (max_iter - 1):
			lam_final = current_lam
			break
		z_old = np.copy(z_new)
		old_lam = np.copy(current_lam)

	return z_new, lam_final


def balancing_principle(A, Y, alpha, noise_std, lam_not = 2.0, q = 0.9):
	"""
	implementation of balancing principle
	"""
	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = noise_std * np.sqrt(m)
	tol = 1e-2

	error = np.zeros(max_iter)
	alt_error = np.zeros(max_iter)
	z_dict = {}
	z_dict[0] = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)

	for i in range(1, max_iter):
		current_lam = lam_not * (q ** i)
		z_dict[i]= fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		error[i] = np.linalg.norm(z_dict[i] - z_dict[i - 1])
		alt_error[i] = np.linalg.norm(z_dict[i] - z_dict[i - 1]) / current_lam

	error = error[ error < (4 * error_level * tol)]
	alt_error = alt_error[ alt_error < (4 * error_level * tol)]
	return current_lam, z_dict[0]

def mc_balancing_principle(A, Y, Y_train, alpha, noise_std, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the so-called monte-carlo balancing principle
	"""
	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = noise_std * np.sqrt(m)
	tol = 1e-2
	kappa = 0.25

	error = np.zeros(max_iter)
	alt_error = np.zeros(max_iter)
	z_dict = {}
	z_dict[0] = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
	rhon = np.zeros(max_iter)
	max_errs = np.zeros(max_iter)
	b_n = np.zeros(max_iter)
	for i in range(1, max_iter):
		current_lam = lam_not * (q ** i)
		z_dict[i]= fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		rhon[i] = np.sqrt(estimate_rhon(A, Y_train, current_lam, alpha))

	for i in range(max_iter):
		error = np.zeros(len(range(i + 1, max_iter)))
		for j in range(i + 1, max_iter):
			if rhon[j] == 0.0:
				error[j - i - 1] = 0
			else:
				error[j - i - 1] = np.linalg.norm(z_dict[j] - z_dict[i]) / (rhon[j] * 4)
		if (i + 1) < max_iter:
			b_n[i] = np.ma.masked_invalid(error).max()

	best_n = len(b_n)
	import pdb; pdb.set_trace()
	for k in range(len(b_n)):
		if (np.max(b_n[k:]) < kappa):
			best_n = k
			break

	best_lam = lam_not * (q ** best_n)
	best_z = z_dict[best_n]
	return best_z, best_lam

def white_noise_balancing_principle(A, Y, alpha, noise_std, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the monte-carlo balancing principle that makes use of the
	white-noise assumption to estimate intrinsic parameters
	"""

	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = noise_std * np.sqrt(m)
	tol = 1e-2
	kappa = 0.25

	error = np.zeros(max_iter)
	alt_error = np.zeros(max_iter)
	z_dict = {}

	rhon = np.zeros(max_iter)

	max_errs = np.zeros(max_iter)
	b_n = np.zeros(max_iter)
	B_n = np.zeros(max_iter)
	for i in range(max_iter):
		current_lam = lam_not * (q ** i)
		z_dict[i]= fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		rhon[i] = noise_std * np.sqrt(white_rhon_sq(A, current_lam, alpha))
	b_n[-1] = np.linalg.norm(z_dict[max_iter - 1] - z_dict[max_iter - 2]) / (rhon[-1] * 4.0)
	B_n[-1] = b_n[-1]
	best_n = 0
	for i in reversed(range(len(b_n) - 1)):
		temp_error = np.zeros(max_iter - 1 - i)
		for j in range(len(temp_error)):
			temp_error[j] = np.linalg.norm(z_dict[i] - z_dict[i + j + 1]) / (rhon[i + j + 1] * 4.0)
		b_n[i] = max(b_n[i + 1], np.max(temp_error))
		B_n[i] = np.max(b_n[i:])
		if B_n[i] > kappa:
			best_n = i + 1
			break

	best_lam = lam_not * (q ** best_n)
	best_z = z_dict[best_n]

	return best_z, best_lam



def hardened_balancing_principle(A, Y, Y_train, alpha, noise_std, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the hardened balancing principle
	"""

	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100
	error_level = noise_std * np.sqrt(m)
	tol = 1e-2
	kappa = 1.0

	error = np.zeros(max_iter)
	alt_error = np.zeros(max_iter)
	z_dict = {}
	z_dict[0] = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
	rhon = np.zeros(max_iter)
	max_errs = np.zeros(max_iter)
	b_n = np.zeros(max_iter)
	for i in range(1, max_iter):
		current_lam = lam_not * (q ** i)
		z_dict[i]= fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		rhon[i] = np.sqrt(estimate_rhon(A, Y_train, current_lam, alpha))

	for i in range(max_iter):
		error = np.zeros(len(range(i + 1, max_iter)))
		for j in range(i + 1, max_iter):
			if rhon[j] == 0.0:
				error[j - i - 1] = 0
			else:
				error[j - i - 1] = np.linalg.norm(z_dict[j] - z_dict[i]) / (rhon[j] * 4)
		if (i + 1) < max_iter:
			b_n[i] = np.ma.masked_invalid(error).max()

	B = np.array([ np.max(b_n[i:]) * np.sqrt(rhon[i]) for i in range(len(b_n) - 1)])
	best_n = np.argmin(B)
	best_lam = lam_not * (q ** best_n)
	best_z = z_dict[best_n]

	return best_z, best_lam

def L_curve(A, Y, alpha, lam_not = 2.0, q = 0.9):
	"""
	Implementation of the L-curve method
	"""
	m = Y.shape[0]
	iter = 0
	current_lam = lam_not * (q ** iter)
	max_iter = 100

	error = np.zeros(max_iter)
	alt_error = np.zeros(max_iter)

	for i in range(max_iter):
		z = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		current_lam = lam_not * (q ** i)
		error[i]  = np.linalg.norm(np.dot(A, z) - Y) * np.linalg.norm(z)

	opt_idx = np.argmin(error)
	lam_final = lam_not * (q ** opt_idx)
	z_final = fixed_point_iterations_lam(A, Y, lam = lam_final, alpha = alpha)
	# print "opt_idx ", opt_idx
	# import pdb; pdb.set_trace()
	return z_final, lam_final

def generalized_cross_validation(A, Y, alpha, lam_not = 2.0, q = 0.9, 	trace_iterations = 10):
	"""
	Implementation of the generalised cross validation, as discussed in Elad
	"""
	max_iter = 100
	m = len(Y)

	error = np.zeros(max_iter)
	trace_estimate = 0.0

	for i in range(max_iter):
		current_lam = lam_not * (q ** i)
		z = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		for j in range(trace_iterations):
			y_trace = np.random.normal(size = m)
			z_trace = fixed_point_iterations_lam(A, y_trace, lam = current_lam, alpha = alpha)
			trace_estimate += np.dot(y_trace, np.dot(A, z_trace))

		trace_estimate *= (1.0 / np.float(trace_iterations))
		error[i]  = (np.linalg.norm(np.dot(A, z) - Y) ** 2 ) / ((1.0 - trace_estimate / np.float(m)) ** 2)

	opt_idx = np.argmin(error)
	lam_final = lam_not * (q ** opt_idx)
	z_final = fixed_point_iterations_lam(A, Y, lam = lam_final, alpha = alpha)
	# print "opt_idx ", opt_idx
	# import pdb; pdb.set_trace()
	return z_final, lam_final

def nonlinear_gcv(A, Y, alpha, lam_not = 2.0, q = 0.9):
	"""
	Implementation of nonlinear generalised cross validation
	"""
	max_iter = 100
	(m, d) = A.shape

	error = np.zeros(max_iter)
	zinv = np.linalg.lstsq(A, Y)[0]

	for i in range(max_iter):
		current_lam = lam_not * (q ** i)
		z = fixed_point_iterations_lam(A, Y, lam = current_lam, alpha = alpha)
		ss = (np.linalg.norm(z, 1) + alpha * np.linalg.norm(z)) / (np.linalg.norm(zinv, 1) + alpha * np.linalg.norm(zinv))

		error[i]  = (np.linalg.norm(np.dot(A, z) - Y) ** 2 ) / (np.float(m) * (1 - ss * np.float(d) / np.float(m)))

	opt_idx = np.argmin(error)
	lam_final = lam_not * (q ** opt_idx)
	z_final = fixed_point_iterations_lam(A, Y, lam = lam_final, alpha = alpha)

	return z_final, lam_final

def armijo_full(A, Y, estimator, alpha_enets, tol = 1e-6, tol2 = 1e-3):
	"""
	Implementation of armijo path line search
	"""

	cnt = 0
	max_iter = 29
	t = np.zeros(max_iter + 1); t[0] = 1.0
	t_current = t[0]
	h = - 0.001
	alpha0 = 0.02
	nu_up = 2
	nu_down = 0.5
	c1 = 0.5

	satisfied = False
	beta = 0.8 # don't know if best
	go = True

	descent_directions = np.zeros(len(t))
	r1 = np.zeros(len(t))
	alphas = np.zeros(len(t))

	# Compute gradient at t
	z = fixed_point_iterations_lam(A, Y, lam = (1 - t_current) / t_current , alpha = alpha_enets)
	t_direction = t_current + h
	z_h = fixed_point_iterations_lam(A, Y, lam = (1 - t_direction) / t_direction , alpha = alpha_enets)
	descent_directions[cnt] =  (np.linalg.norm(z - estimator) - np.linalg.norm(z_h - estimator)) / h # p_k
	r1[cnt] =  np.linalg.norm(z - estimator) #r1
	t_temp = t_current + alpha0 * descent_directions[cnt]
	r2 = np.linalg.norm(fixed_point_iterations_lam(A, Y, lam = (1 - t_temp) / t_temp , alpha = alpha_enets) - estimator)

	while satisfied == False:
		# Check stopping criteria
		if (np.abs(descent_directions[cnt]) < tol) or (cnt >= max_iter):
			satisfied = True
			break
		# Find the step-size
		alpha = np.copy(alpha0)
		tau = - np.abs(descent_directions[cnt]) ** 2 #phiprime0
		in_cnt = 0
		condition = False

		while condition == False:
			phi1 = np.copy(r1[cnt])
			phiprime1 = np.copy(tau)
			phi2 = np.copy(r2)
			alpha_min = - (alpha ** 2 * phiprime1) / (2.0 * (phi2 - phi1 - phiprime1 * alpha))
			previous_alpha = np.copy(alpha)

			if (alpha_min <= (nu_up * alpha)) and (alpha_min >= (nu_down * alpha)):
				alpha = alpha_min
			elif (alpha_min > (nu_up * alpha)):
				alpha = nu_up * alpha
			elif (alpha_min < (nu_down * alpha)):
				alpha = nu_down * alpha

			t_temp = t_current + alpha * descent_directions[cnt]
			r2 = np.linalg.norm(fixed_point_iterations_lam(A, Y, lam = (1 - t_temp) / t_temp , alpha = alpha_enets) - estimator)
			if (np.abs(phi2 - phi1) < (c1 * phiprime1 * alpha)) or (np.abs(alpha - previous_alpha) < tol2) or (in_cnt > max_iter):
				step_size = alpha
				condition = True
			in_cnt += 1

		t_new = t_current + step_size * descent_directions[cnt]
		t[cnt + 1] = np.copy(t_new)
		t_current = np.copy(t_new)
		# Compute new stuff

		z = fixed_point_iterations_lam(A, Y, lam = (1 - t_current) / t_current , alpha = alpha_enets)
		t_direction = t_current + h
		z_h = fixed_point_iterations_lam(A, Y, lam = (1 - t_direction) / t_direction , alpha = alpha_enets)
		descent_directions[cnt + 1] =  (np.linalg.norm(z - estimator) - np.linalg.norm(z_h - estimator)) / h # p_k
		r1[cnt + 1] =  np.linalg.norm(z - estimator) #r1
		t_temp = t_current + alpha0 * descent_directions[cnt]
		r2 = np.linalg.norm(fixed_point_iterations_lam(A, Y, lam = (1 - t_temp) / t_temp , alpha = alpha_enets) - estimator)
		cnt += 1

	return t[cnt]

def backtracking_interpolate(A, Y, estimator, alpha_enets, tol = 1e-6):
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
		z = fixed_point_iterations_lam(A, Y, lam = (1 - t_current) / t_current , alpha = alpha_enets)
		t_direction = t_current + h
		z_h = fixed_point_iterations_lam(A, Y, lam = (1 - t_direction) / t_direction , alpha = alpha_enets)
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
		phialpha = np.linalg.norm(fixed_point_iterations_lam(A, Y, lam = (1 - t_temp) / t_temp , alpha = alpha_enets) - estimator)

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

def backtracking_interpolate_sq(A, Y, estimator, alpha_enets, tol = 1e-6):
	"""
	Implementation of the backtracking line search for the squared error function
	ie for |z_old-z_new|^2, instead of |z_old-z_new|, as in the previous code
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
	z_h = np.zeros_like(estimator)

	while satisfied == False:
		# Compute gradient at t
		cnt += 1
		z = fixed_point_iterations_lam(A, Y, lam = (1 - t_current) / t_current , alpha = alpha_enets, z = z_h)
		t_direction = t_current + h
		z_h = fixed_point_iterations_lam(A, Y, lam = (1 - t_direction) / t_direction , alpha = alpha_enets, z = z)

		descent_directions[cnt] =  (np.linalg.norm(z - estimator) ** 2 - np.linalg.norm(z_h - estimator) ** 2) / h
		loss_functional_evals[cnt] =  np.linalg.norm(z - estimator) ** 2

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
		phialpha = np.linalg.norm(fixed_point_iterations_lam(A, Y, lam = (1 - t_temp) / t_temp , alpha = alpha_enets, z = z_h) - estimator) ** 2

		if (phialpha - phi0) < (c1 * phiprime0 * alpha):
			step_size = alpha
		else:
			step_size = - (alpha ** 2 * phiprime0) / (2.0 * (phialpha - phi0 - phiprime0 * alpha))
		if (np.abs(last_alpha / step_size) > 10.0) or (np.abs(step_size) < 1e-4):
			step_size = last_alpha * beta * 1.3

		last_alpha = np.copy(step_size)
		t_new = t_current + step_size * descent_directions[cnt]
		t[cnt + 1] = t_new
		t_current = t_new

	return t_new


def grid_search_my(A, Y, estimator, alpha_enets, t_all = None):
	"""
	Run a grid search on the range of parameters t_all
	To do so	- create an error vector corresponding to each t in t_all
				- compute a solution for each t in t_all
				- return the t with the smallest error wrt estimator
				- estimator can be the true solution, the empirical estimator
					or whatever else
	"""
	if t_all is None:
	    t_all = np.linspace(0.01, 1, 200)

	Apinv = np.linalg.pinv(A)
	P = np.dot(Apinv, A)
	error = np.zeros(len(t_all))
	z = np.zeros_like(estimator)
	for i in range(len(t_all)):
	    z = fixed_point_iterations_lam(A, Y, lam = (1 - t_all[i]) / t_all[i] , alpha = alpha_enets, z = z)
	    error[i] = np.linalg.norm(z - estimator)
	t_orig = t_all[np.argmin(error)]

	return t_orig

if __name__ == "__main__":
	### An example use of the code
	m = 100
	d = 100
	# create data
	A, piY_n, X_test, Y_test = set_it_up(m, d, rank = 100, h = 20, N_train = 100, N_test = 1, sigma = 0.3)
        print(X_test)
        print(Y_test)
	# create the empirical estimator
	x_hat = np.dot(np.linalg.pinv(A), np.dot(piY_n, Y_test.T))
	t_bt = backtracking_interpolate(A, Y_test.T, x_hat, 1e-5)
	#impori pdb; pdb.set_trace()
