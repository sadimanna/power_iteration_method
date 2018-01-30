import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as la
#from sksparse.cholmod import cholesky

def inv_power_iter(a,niter=1000):
	tol  = 10**(-9)
	ashape = a.get_shape()
	#initializing the eigenvector
	v = np.random.random(ashape[0])
	#pre-normalize
	v = v/np.linalg.norm(v)
	#eigenvalue
	lo = np.dot(np.transpose(v),a.dot(v))/np.dot(np.transpose(v),v)
	for _ in xrange(niter):
		#solve for w instead of calculating the inverse of A
		w = la.spsolve(a,v)
		#re normalze
		v = w/np.linalg.norm(w)
		#calculating eigenvalue
		l = np.dot(np.transpose(v),a.dot(v))/np.dot(np.transpose(v),v)
		#Check if satisfies convergence criteria
		if abs(l-lo)/l < tol:
			return v, l
		lo = l
	return v, l

def get_mineigval(A):
	niter = 200000
	return inv_power_iter(A,niter)

def get_shifted_eigvec(A,s,eigval):
	ashape = A.get_shape()
	rc = np.arange(ashape[0])
	I = sp.coo_matrix((np.ones(ashape[0]),(rc,rc)),shape = ashape)
	#apply shift
	Atemp = A - s*I
	niter = 200000
	v, l = inv_power_iter(Atemp,niter)
	return v, l

