import numpy as np
import scipy.sparse as sp
import time

def power_iteration(A, num_simulations):
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
		
		eigval_new = np.dot(np.transpose(eigvec),A.dot(eigvec))/np.dot(np.transpose(eigvec),eigvec)
		if (abs(eigval_new-eigval_old)/eigval_new) < tol:
			return eigval_new
		eigval_old = eigval_new
		
	return eigval_new

def get_maxeigval(A):
	niter = 200000
	#maxeigval = power_iteration(A,niter)
	#UNSTABLE
	#rc = np.array([i for i in xrange(Ashape[0])])
	#Atemp = A - maxeigval*sp.coo_matrix((np.ones(Ashape[0]),(rc,rc)),shape=Ashape)
	#mineigval = power_iteration(Atemp,niter) + maxeigval

	return power_iteration(A,niter)

#stime=time.time()
#A = sp.load_npz('ham_kin.npz')
#print 'Time to load..'+str(time.time()-stime)+' seconds...'
#print 'Total time...'+str(time.time()-stime)+' seconds...'
