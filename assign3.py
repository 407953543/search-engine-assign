import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt

#task1
A = np.array([[1,0,1,0,1,1,1,1,1,0,0],[1,1,0,1,0,0,1,1,0,2,1],[1,1,0,0,0,1,1,1,1,0,1]])
query = np.array([0,0,0,0,0,1,0,0,0,1,1])
U,sigma,VT = la.svd(A.T)

print('A=')
print(A.T)
print('U=')
print(U)
print('sigma=')
print(sigma)
print('VT=')
print(VT)

#task2
sig3 = np.mat(np.eye(3)*sigma[:3])
innerProduct1 = np.dot(A[0],query)
innerProduct2 = np.dot(A[1],query)
innerProduct3 = np.dot(A[2],query)

print('newA=')
print(np.dot(np.dot(U[:,:3],sig3),VT))
print('innerProduct1=')
print(innerProduct1)
print('innerProduct2=')
print(innerProduct2)
print('innerProduct3=')
print(innerProduct3)

#task3
U2 = U[:,:2]
sig2 = np.mat(np.eye(2)*sigma[:2])
V2T = VT[:2][:]
V2 = V2T.T

print('U2=')
print(U2)
print('sig2=')
print(sig2)
print('V2T=')
print(V2T)
print('V2=')
print(V2)
print('A2=')
print(np.dot(U2,np.dot(sig2,V2T)))

#task4
A2 = np.dot(sig2,V2T)
q2 = np.dot(np.dot(sig2.I,U2.T),query.T).T
A2T = A2.T

print('d1=')
print(A2T[0])
print('d2=')
print(A2T[1])
print('d3=')
print(A2T[2])
print('q2=')
print(q2.T)
plt.scatter(A2[0].tolist(),A2[1].tolist())
plt.scatter(q2[0].tolist(),q2[1].tolist())
plt.show()

#task5
newProduct1 = np.dot(A2T[0],q2)
newProduct2 = np.dot(A2T[1],q2)
newProduct3 = np.dot(A2T[2],q2)

print('newProduct1=')
print(newProduct1)
print('newProduct2=')
print(newProduct2)
print('newProduct3=')
print(newProduct3)