import numpy as np    
import cv2    
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
import matplotlib.pyplot as plt
from skimage import io, color

#Loading original image
originImg = cv2.imread('3.jpg')
img_lab = cv2.cvtColor(originImg, cv2.COLOR_BGR2LAB)
img_rgb = cv2.cvtColor(originImg, cv2.COLOR_BGR2RGB)

# Shape of original image    
originShape = originImg.shape


# Converting image into array of dimension [nb of pixels in originImage, 3]
# based on r g b intensities    
flatImg1=np.reshape(img_lab, [-1, 3])
flatImg2=np.copy(np.reshape(img_lab, [-1, 3]))


# Estimate bandwidth for meanshift algorithm    
#bandwidth = estimate_bandwidth(flatImg1, quantile=0.1, n_samples=100)    
#print(bandwidth)
ms = MeanShift(bandwidth = 15, bin_seeding=True)
# Performing meanshift on flatImg    
ms.fit(flatImg1)
# (r,g,b) vectors corresponding to the different clusters after meanshift    
labels=ms.labels_
# Remaining colors after meanshift    
cluster_centers = ms.cluster_centers_ 
# Finding and diplaying the number of clusters    
labels_unique = np.unique(labels)    
n_clusters_ = len(labels_unique)    
print("number of estimated clusters : %d" % n_clusters_)    

#K Means clustering
km = KMeans(n_clusters=n_clusters_)
km.fit(flatImg2)
labels_k=km.labels_
cluster_centers_k = km.cluster_centers_   

# Displaying segmented image    
segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
segmentedImg_k = cluster_centers_k[np.reshape(labels_k, originShape[:2])]
#l = segmentedImg[:,:,0]
#a = segmentedImg[:,:,1]
#b = segmentedImg[:,:,2]

seg = cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_LAB2RGB)
seg_k = cv2.cvtColor(segmentedImg_k.astype(np.uint8), cv2.COLOR_LAB2RGB)
#seg_rgb = cv2.cvtColor(segmentedImg.astype(np.uint8), cv2.COLOR_BGR2RGB)
#cv2.imshow('Image',segmentedImg.astype(np.uint8))

#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.subplot(131), plt.imshow(img_rgb)
plt.title('original'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(seg)
#plt.subplot(122), plt.imshow(np.reshape(labels, segmentedImg.shape[:2]))
plt.title('Mean shift'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(seg_k)
plt.title('K means'), plt.xticks([]), plt.yticks([])

plt.show()