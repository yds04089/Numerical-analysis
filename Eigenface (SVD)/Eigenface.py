import numpy as np
import os
import glob
import shutil
from PIL import Image, ImageDraw
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.spatial import distance
#from sklearn.decomposition.pca import PCA

path_name = "/Users/yds04/Desktop/3학년 2학기/수치해석/Project1/dataset/"
#images_list = glob.glob(path_name + "*.pgm")
images_list_ = glob.glob(path_name + "*_half.pgm")

M = np.zeros((32, 32))
A = np.zeros((1, 1024), float)


for i in images_list_:
    im = Image.open(i)
    pix = np.array(im)
    #plt.imshow(pix,cmap = 'gray')
#plt.imshow(np.reshape(ans1, (32, 32)))
    #plt.show()
    M+=pix

M = M/float(len(images_list_))
print(M)

for i in images_list_:
    im = Image.open(i)
    pix = np.array(im)
    pix = np.subtract(pix, M)
    pix_ = np.reshape(pix, (1, 1024))
    A = np.append(A, pix_, axis = 0) 

A = np.delete(A, [0, 0], axis = 0)
#plt.imshow(np.reshape(ans1, (32, 32)))
A_ = np.dot(A, A.T)
#print(A.shape)
#U, s, V = np.linalg.svd(A, full_matrices=False)
#pca = PCA(n_components=50)
#pca.fit(A)

num_components = 100
U_, s_, V_ = svds(A.T, k=num_components)

#basis = U[0:50].copy()
basis = np.array(U_.T)
print("V: ", V_.shape)
print("U: ", U_.shape)
print(s_)
print(basis.shape)
print(basis)
print("/////////////")
#for i in range(0, 50):
 #   print(np.linalg.norm(basis[i]))

#print(basis[1].shape)
C1 = np.zeros(100)
C2 = np.zeros(100)
C3 = np.zeros(100)
C4 = np.zeros(100)
C5 = np.zeros(100)
C6 = np.zeros(100)
C7 = np.zeros(100)
C8 = np.zeros(100)
C9 = np.zeros(100)

for i in range (0, 100):
    C1[i] = np.dot(A[3], basis[i].T)
    C2[i] = np.dot(A[4], basis[i].T)
    C3[i] = np.dot(A[5], basis[i].T)
    C4[i] = np.dot(A[6], basis[i].T)
    C5[i] = np.dot(A[7], basis[i].T)
    C6[i] = np.dot(A[10], basis[i].T)
    C7[i] = np.dot(A[20], basis[i].T)
    C8[i] = np.dot(A[30], basis[i].T)
    C9[i] = np.dot(A[50], basis[i].T)
#print(C1)
#print(C1.shape)
ans1 = np.zeros(1024)
ans2 = np.zeros(1024)
ans3 = np.zeros(1024)
ans4 = np.zeros(1024)
ans5 = np.zeros(1024)
ans6 = np.zeros(1024)
ans7 = np.zeros(1024)
ans8 = np.zeros(1024)
ans9 = np.zeros(1024)
for i in range (0, 100):
    ans1+=np.dot(C1[i], basis[i])
    ans2+=np.dot(C2[i], basis[i])
    ans3+= np.dot(C3[i], basis[i])
    ans4+=np.dot(C4[i], basis[i])
    ans5+= np.dot(C5[i], basis[i])
    ans6+= np.dot(C6[i], basis[i])
    ans7+= np.dot(C7[i], basis[i])
    ans8+= np.dot(C8[i], basis[i])
    ans9+= np.dot(C9[i], basis[i])


print("same")
print(distance.euclidean(C1, C2))
print(distance.euclidean(C1, C3))
print(distance.euclidean(C1, C4))
print(distance.euclidean(C2, C4))
print("different")
print(distance.euclidean(C1, C5))
print(distance.euclidean(C1, C6))
print(distance.euclidean(C1, C7))
print(distance.euclidean(C1, C8))
print(distance.euclidean(C1, C9))



