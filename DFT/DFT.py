import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import reshape
from scipy.spatial import distance

distlist = []
CArr = np.empty((159,), dtype = float)

def fourier(name):
   global distlist
   global CArr
   img_ = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
   print(name)
   img__=cv2.resize(img_, dsize=(128,128))
   img1=img__[32:96,32:96]
   f1=np.fft.fft2(img1)
   fshift1=np.fft.fftshift(f1)
   magnitude1=32*np.log(np.abs(fshift1))
   magnitude1[32][32] = 0

   C1 = magnitude1[0:33, 0:33]
   arr1 = np.array([])
   for i in range(31):
      for j in range(33):
         if j in [31, 32]:
            arr1 = np.append(arr1, C1[i][j])
         elif i == j:
            arr1 = np.append(arr1, C1[i][j])
            
   for i in [31, 32]:
      arr1 = np.append(arr1, C1[i][:])
   print(arr1.shape)

   img2=img__[1:65,1:65]
   f2=np.fft.fft2(img2)
   fshift2=np.fft.fftshift(f2)
   magnitude2=32*np.log(np.abs(fshift2))
   magnitude2[32][32] = 0

   C2 = magnitude2[0:33, 0:33]
   arr2 = np.array([])
   for i in range(31):
      for j in range(33):
         if j in [31, 32]:
            arr2 = np.append(arr2, C2[i][j])
         elif i == j:
            arr2 = np.append(arr2, C2[i][j])
   for i in [31, 32]:
      arr2 = np.append(arr2, C2[i][:])
   print(arr2.shape)


   img3=img__[1:65,63:127]
   f3=np.fft.fft2(img3)
   fshift3=np.fft.fftshift(f3)
   magnitude3=32*np.log(np.abs(fshift3))
   magnitude3[32][32] = 0

   C3 = magnitude3[0:33, 0:33]
   arr3 = np.array([])
   for i in range(31):
      for j in range(33):
         if j in [31, 32]:
            arr3 = np.append(arr3, C3[i][j])
         elif i == j:
            arr3 = np.append(arr3, C3[i][j])
   for i in [31, 32]:
      arr3 = np.append(arr3, C3[i][:])

   img4=img__[63:127,1:65]
   f4=np.fft.fft2(img4)
   fshift4=np.fft.fftshift(f4)
   magnitude4=32*np.log(np.abs(fshift4))
   magnitude4[32][32] = 0

   C4 = magnitude4[0:33, 0:33]
   arr4 = np.array([])
   for i in range(31):
      for j in range(33):
         if j in [31, 32]:
            arr4 = np.append(arr4, C4[i][j])
         elif i == j:
            arr4 = np.append(arr4, C4[i][j])
            
   for i in [31, 32]:
      arr4 = np.append(arr4, C4[i][:])

   img5=img__[63:127,63:127]
   f5=np.fft.fft2(img5)
   fshift5=np.fft.fftshift(f5)
   magnitude5=32*np.log(np.abs(fshift5))
   magnitude5[32][32] = 0

   C5 = magnitude5[0:33, 0:33]
   arr5 = np.array([])
   for i in range(31):
      for j in range(33):
         if j in [31, 32]:
            arr5 = np.append(arr5, C5[i][j])
         elif i == j:
            arr5 = np.append(arr5, C5[i][j])
   for i in [31, 32]:
      arr5 = np.append(arr5, C5[i][:])

   M = (magnitude1+magnitude2+magnitude3+magnitude4+magnitude5)/5
   #print(M)
   C = M[0:33, 0:33]
   arr = np.array([])
   for i in range(31):
      for j in range(33):
         if j in [31, 32]:
            arr = np.append(arr, C[i][j])
         elif i == j:
            arr = np.append(arr, C[i][j])
   for i in [31, 32]:
      arr = np.append(arr, C[i][:])

   dist1 = np.linalg.norm(arr - arr1)
   dist2 = np.linalg.norm(arr - arr2)
   dist3 = np.linalg.norm(arr - arr3)
   dist4 = np.linalg.norm(arr - arr4)
   dist5 = np.linalg.norm(arr - arr5)
   #print(max(dist1, dist2, dist3, dist4, dist5))
   distlist.append(max(dist1, dist2, dist3, dist4, dist5))
   CArr = np.append(CArr, arr, axis = 0)
   #Carr = CArr.reshape(20, -1)
   #print(CArr.shape)
   #print(Carr.shape)
   #print(arr.shape)
   #print(arr)
   #rows,cols=img.shape
   #crow,ccol=int(rows/2), int(cols/2)

   #fshift[crow-30:crow+30, ccol-30:ccol+30]=0
   #f_ishift1=np.fft.ifftshift(fshift1)
   #img_back1=np.fft.ifft2(f_ishift1)
   #img_back1=np.abs(img_back1) 

   #plt.subplot(131), plt.imshow(img1, cmap='gray')
   #plt.title('original'), plt.xticks([]), plt.yticks([])

  # plt.subplot(132), plt.imshow(img_back, cmap='gray')
   #plt.title('after'), plt.xticks([]), plt.yticks([])

  # plt.subplot(133), plt.imshow(magnitude1, cmap='gray')
  # plt.title('magnitude'), plt.xticks([]), plt.yticks([])

   #plt.show()

def test(name):
   global distlist
   global CArr, Carr
   global maxdist
   img_ = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
   print(name)
   img__=cv2.resize(img_, dsize=(128,128))
   img1=img__[44:108,25:89]
   f1=np.fft.fft2(img1)
   fshift1=np.fft.fftshift(f1)
   magnitude1=32*np.log(np.abs(fshift1))
   magnitude1[32][32] = 0

   C1 = magnitude1[0:33, 0:33]
   arr1 = np.array([])
   for i in range(31): 
      for j in range(33):
         if j in [31, 32]:
            arr1 = np.append(arr1, C1[i][j])
         elif i == j:
            arr1 = np.append(arr1, C1[i][j])
   for i in [31, 32]:
      arr1 = np.append(arr1, C1[i][:])
   print(arr1.shape)
   print()
   cnt = 0
   possible = []
   pidx = []
   for i in range(20):
      dist = np.linalg.norm(Carr[i][:] - arr1)
      if dist <= maxdist:
         possible.append(dist)
         pidx.append(i)
         cnt+=1
   if cnt == 0:
      print("NO")
   else:
      max = 0
      ansidx = 0
      for idx, val in enumerate(possible):
         if val > max:
            max = val
            ansidx = idx
      print("ANS: ", pidx[ansidx]+1, max)      



path_dir='Input'
file_list=os.listdir(path_dir)
print(file_list)
for i in range(20):
   str='Input/'+file_list[i]
   fourier(str)
Carr = CArr[159:]
print(Carr.shape)
Carr = Carr.reshape(20, -1)
print(Carr.shape)
print(distlist)
maxdist = max(distlist)
print("MAX", max(distlist))
print("TEST")
for i in range(30):
   str='Input/'+file_list[i]
   test(str)


