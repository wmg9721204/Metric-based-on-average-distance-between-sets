import numpy as np
from metrics import avg_dist_normalize2

n = 200
thetas = [2*np.pi/n*k for k in range(n)]
X = np.array([[np.cos(theta), np.sin(theta)] for theta in thetas])
Y = 0.001*np.array([[np.cos(theta), np.sin(theta)] for theta in thetas])
# Y = np.random.random((1000,2))
X = np.random.random((1000,2))-1/2
Y = 2*X

## sanity check
print(avg_dist_normalize(X,Y), avg_dist_normalize(X,X))
print(avg_dist_normalize2(X,Y), avg_dist_normalize2(X,X))

X0 = X[:,0].reshape(-1,1)
X1 = X[:,1].reshape(-1,1)

Y0 = Y[:,0].reshape(-1,1)
Y1 = Y[:,1].reshape(-1,1)

## test for dimensional additivity

feat_wise_sum = avg_dist_normalize2(X0, Y0)+avg_dist_normalize2(X1, Y1)
direct_compute = avg_dist_normalize2(X,Y)

if abs(feat_wise_sum-direct_compute)<1e-10:
    print('Dimensional additivity passed.')
else:
    print('Dimensional additivity failed.')
