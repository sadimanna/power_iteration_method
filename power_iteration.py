import numpy as np
import scipy.sparse as sp
import time

def power_iteration(A, niter):
	tol = 10**(-9)
	Ashape = A.get_shape()
        eigvec = np.random.rand(Ashape[0])
	eigval_old = np.dot(np.transpose(eigvec),A.dot(eigvec))/np.dot(np.transpose(eigvec),eigvec)        
	for i in range(num_simulations):
        	# calculate the matrix-by-vector product Ab
        	eigvec1 = A.dot(eigvec)
        	# calculate the norm
        	eigvec1_norm = np.linalg.norm(eigvec1)
        	# re normalize the vector
        	eigvec = eigvec1 / eigvec1_norm
		#eigenvalue
		eigval_new = np.dot(np.transpose(eigvec),A.dot(eigvec))/np.dot(np.transpose(eigvec),eigvec)
		if (abs(eigval_new-eigval_old)/eigval_new) < tol:
			return eigval_new
		eigval_old = eigval_new
		
	return eigval_new

def get_maxeigval(A):
	niter = 200000
	return power_iteration(A,niter)
