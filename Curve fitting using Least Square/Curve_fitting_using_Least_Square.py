import numpy as np 
import random
from matplotlib import pyplot as plt 

def f1(a):
	return X1[0]*a*a+X1[1]*a+X1[2]

def f2(a):
	return X2[0]*a*a+X2[1]*a+X2[2]

data = np.array([[-2.9, 35.4], 
		[-2.1, 19.7], 
		[-0.9, 5.7], 
		[1.1, 2.1], 
		[0.1, 1.2], 
		[1.9, 8.7],
		[3.1, 25.7], 
		[4.0, 41.5]])

data1 = random.sample([0, 1, 2, 3, 4, 5, 6, 7], 6) #random하게 점 6개를 뽑음
data2 = random.sample([0, 1, 2, 3, 4, 5, 6, 7], 6)

print('data1: ', data1)
print('data2: ', data2)

A1 = np.ones((6, 3))
idx = 0
for i in data1:
	A1[idx, 0] = data[i, 0]**2
	A1[idx, 1] = data[i, 0]
	A1[idx, 2] = data[i, 0]
	idx += 1
X1 = np.dot(np.linalg.pinv(A1), data[data1[:], 1])

A2 = np.ones((6, 3))
idx = 0
for i in data2:
	A2[idx, 0] = data[i, 0]**2
	A2[idx, 1] = data[i, 0]
	A2[idx, 2] = data[i, 0]
	idx += 1
X2 = np.dot(np.linalg.pinv(A2), data[data2[:], 1])

print('A1: ', A1)
print('A2: ', A2)

print('X1: ', X1)
print('X2: ', X2)

x1 = np.arange(-10, 10, 0.01)
x2 = np.arange(-10, 10, 0.01)

plt.scatter(data[:, 0], data[:, 1]) 	#그래프를 그림
plt.title("Curve fitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis([-7, 7, -10, 50])
plt.plot(x1, f1(x1), x2, f2(x2), 'r-')
plt.show()