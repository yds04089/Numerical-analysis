import cv2
import numpy as np
import math
import os
from matplotlib import pyplot as plt
import pandas as pd

def correlation(imgname):
   img_ = cv2.imread(imgname)
   img__=cv2.resize(img_, dsize=(512,512))
   yuv_img = cv2.cvtColor(img__, cv2.COLOR_BGR2YUV)
   b, g, r = cv2.split(img__)
   y, u, v = cv2.split(yuv_img)
   zero = y.copy()
   zero[:,:] = 128
   img = cv2.merge([r,g,b])
   imgR = img.copy()
   imgR[:,:,1] = 0
   imgR[:,:,2] = 0
   imgG = img.copy()
   imgG[:,:,0] = 0
   imgG[:,:,2] = 0
   imgB = img.copy()
   imgB[:,:,0] = 0
   imgB[:,:,1] = 0

   imgY = yuv_img.copy()
   imgY[:,:,1] = 0
   imgY[:,:,2] = 0
   imgU = yuv_img.copy()
   imgU[:,:,0] = 0
   imgU[:,:,2] = 0
   imgV = yuv_img.copy()
   imgV[:,:,0] = 0
   imgV[:,:,1] = 0
   
   imgY = cv2.merge([y,y,y])
   imgU = cv2.merge([zero,zero,u])
   imgV = cv2.merge([v,zero,zero])

   imgU_ = cv2.cvtColor(imgU, cv2.COLOR_YUV2RGB)
   imgY_ = cv2.cvtColor(imgY, cv2.COLOR_YUV2RGB)
   imgV_ = cv2.cvtColor(imgV, cv2.COLOR_YUV2RGB)

   r_ = r.reshape(1, -1)
   g_ = g.reshape(1, -1)
   b_ = b.reshape(1, -1)
   RG = np.cov(r_,g_)
   GB = np.cov(g_,b_)
   BR = np.cov(b_,r_)
   cor_RG = RG[0][1]/(math.sqrt(RG[0][0])*math.sqrt(RG[1][1]))
   cor_GB = GB[0][1]/(math.sqrt(GB[0][0])*math.sqrt(GB[1][1]))
   cor_BR = BR[0][1]/(math.sqrt(BR[0][0])*math.sqrt(BR[1][1]))

   y_ = y.reshape(1, -1)
   u_ = u.reshape(1, -1)
   v_ = v.reshape(1, -1)
   YU = np.cov(y_,u_)
   UV = np.cov(u_,v_)
   VY = np.cov(v_,y_)
   cor_YU = YU[0][1]/(math.sqrt(YU[0][0])*math.sqrt(YU[1][1]))
   cor_UV = UV[0][1]/(math.sqrt(UV[0][0])*math.sqrt(UV[1][1]))
   cor_VY = VY[0][1]/(math.sqrt(VY[0][0])*math.sqrt(VY[1][1]))
   
   print(imgname)
   print("RG: ", cor_RG, " | GB: ", cor_GB, " | BR: ", cor_BR)
   print("YU: ", cor_YU, " | UV: ", cor_UV, " | VY: ", cor_VY)

   plt.subplot(241), plt.imshow(img)
   plt.title('original'), plt.xticks([]), plt.yticks([])

   plt.subplot(242), plt.imshow(imgR)
   plt.title('R'), plt.xticks([]), plt.yticks([])

   plt.subplot(243), plt.imshow(imgG)
   plt.title('G'), plt.xticks([]), plt.yticks([])

   plt.subplot(244), plt.imshow(imgB)
   plt.title('B'), plt.xticks([]), plt.yticks([])

   plt.subplot(245), plt.imshow(img)
   plt.title('original'), plt.xticks([]), plt.yticks([])

   plt.subplot(246), plt.imshow(imgV)
   plt.title('V'), plt.xticks([]), plt.yticks([])

   plt.subplot(247), plt.imshow(imgY)
   plt.title('Y'), plt.xticks([]), plt.yticks([])

   plt.subplot(248), plt.imshow(imgU)
   plt.title('U'), plt.xticks([]), plt.yticks([])
   
   plt.show()
   


path_dir='Input'
file_list=os.listdir(path_dir)
for i in range(10):
   str='Input/'+ file_list[i]
   correlation(str)
