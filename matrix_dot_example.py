import numpy as np

W = np.array([[1,2],[3,4],[5,6]])
print(W.shape)
print(W)

X = np.array([7,8])
print(X.shape)
print(X)

Y=np.dot(W,X)
print(Y)