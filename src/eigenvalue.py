import numpy as np
from numpy import linalg

def power_iteration(A, v0, max_iter=50, tol=1e-2):
    vec = v0/np.linalg.norm(v0, ord=2)
    eigenvalue = 0
    for it in range(max_iter):
        print("Starting Eigenvalue Iteration: ", it)
        w = np.matmul(A, vec)
        vec_new = w/np.linalg.norm(w, ord=2)
        eigenvalue_new = np.matmul(vec.transpose(), np.matmul(A, vec))
        res = np.abs(eigenvalue_new - eigenvalue)
        print("Norm: ", res)
        if res < tol:
            break
        vec = vec_new
        eigenvalue = eigenvalue_new
    return eigenvalue_new, vec_new
