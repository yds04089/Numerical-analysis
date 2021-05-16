import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from scipy.spatial import distance
#from sklearn.preprocessing import StandardScaler

x1 = np.random.normal(0, 5, 300)
y1 = np.random.normal(10, 3, 300)
z1 = np.random.normal(30, 6, 300)

x2 = np.random.normal(8, 5, 300)
y2 = np.random.normal(12, 3, 300)
z2 = np.random.normal(10, 6, 300)

x3 = np.random.normal(10, 3, 300)
y3 = np.random.normal(3, 4, 300)
z3 = np.random.normal(0, 3, 300)

x4 = np.random.normal(5, 2, 300)
y4 = np.random.normal(-10, 5, 300)
z4 = np.random.normal(20, 6, 300)

x5 = np.random.normal(-10, 3, 300)
y5 = np.random.normal(20, 5, 300)
z5 = np.random.normal(15, 5, 300)

X = np.concatenate((x1, x2, x3, x4, x5), axis = None)
Y = np.concatenate((y1, y2, y3, y4, y5), axis = None)
Z = np.concatenate((z1, z2, z3, z4, z5), axis = None)

k = np.array((X, Y, Z))
data = k.T
print(data.shape)
print(data[1])

k1 = np.array((x1, y1, z1))
data1 = k1.T
k2 = np.array((x2, y2, z2))
data2 = k2.T
k3 = np.array((x3, y3, z3))
data3 = k3.T
k4 = np.array((x4, y4, z4))
data4 = k4.T
k5 = np.array((x5, y5, z5))
data5 = k5.T

#KMeans Clustering
n_clusters_ = 5
km = KMeans(n_clusters=n_clusters_)
km.fit(data)
labels_p = km.predict(data)
labels=km.labels_
cluster_centers_k = km.cluster_centers_  
print(cluster_centers_k)
print(labels)
print(labels_p)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
cluster_centers_x = cluster_centers_k[:, 0]
cluster_centers_y = cluster_centers_k[:, 1]
cluster_centers_z = cluster_centers_k[:, 2]
#print(np.mean(x1))

# learning
min_list = [0, 0, 0, 0, 0]
cnt_list = [0, 0, 0, 0, 0]
max_ = 0
sum1 = 0
sum2 = 0
m2 = 0
m1 = 0
for i in range(300):
    list = []
    list.append([distance.euclidean(data1[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data1[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data1[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data1[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data1[i], cluster_centers_k[4]), 4])
    list.sort()
    cnt_list[list[0][1]]+=1
    sum1+=list[0][0]
    sum2+=list[1][0]
    #print(list[0], list[4])
print("data1: ",cnt_list, sum1/300, sum2/300)
m2 += sum2/300
m1 += sum1/300
min_list[0] = (sum1/300+ sum2/300)/2

cnt_list = [0, 0, 0, 0, 0]
sum1 = 0
sum2 = 0
for i in range(300):
    list = []
    list.append([distance.euclidean(data2[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data2[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data2[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data2[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data2[i], cluster_centers_k[4]), 4])
    list.sort()
    cnt_list[list[0][1]]+=1
    sum1+=list[0][0]
    sum2+=list[1][0]
    #print(list[0], list[4])
print("data2: ",cnt_list, sum1/300, sum2/300)
m2 += sum2/300
m1 += sum1/300
min_list[1] = (sum1/300+ sum2/300)/2

cnt_list = [0, 0, 0, 0, 0]
sum1 = 0
sum2 = 0
for i in range(300):
    list = []
    list.append([distance.euclidean(data3[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data3[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data3[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data3[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data3[i], cluster_centers_k[4]), 4])
    list.sort()
    cnt_list[list[0][1]]+=1
    sum1+=list[0][0]
    sum2+=list[1][0]
    #print(list[0], list[4])
print("data3: ",cnt_list, sum1/300, sum2/300)
m2 += sum2/300
m1 += sum1/300
min_list[2] = (sum1/300+ sum2/300)/2

cnt_list = [0, 0, 0, 0, 0]
sum1 = 0
sum2 = 0
for i in range(300):
    list = []
    list.append([distance.euclidean(data4[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data4[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data4[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data4[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data4[i], cluster_centers_k[4]), 4])
    list.sort()
    cnt_list[list[0][1]]+=1
    sum1+=list[0][0]
    sum2+=list[1][0]
    #print(list[0], list[4])
print("data4: ",cnt_list, sum1/300, sum2/300)
m2 += sum2/300
m1 += sum1/300
min_list[3] = (sum1/300+ sum2/300)/2

cnt_list = [0, 0, 0, 0, 0]
sum1 = 0
sum2 = 0
for i in range(300):
    list = []
    list.append([distance.euclidean(data5[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data5[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data5[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data5[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data5[i], cluster_centers_k[4]), 4])
    list.sort()
    cnt_list[list[0][1]]+=1
    sum1+=list[0][0]
    sum2+=list[1][0]
    #print(list[0], list[4])
print("data5: ",cnt_list, sum1/300, sum2/300)
m2 += sum2/300
m1 += sum1/300
min_list[4] = (sum1/300+ sum2/300)/2

m2 /= 5
m1 /= 5
m2 = (m1+m2)/2
print(m2)
print(min_list)


#test
print("test")
x1_ = np.random.normal(0, 5, 100)
y1_ = np.random.normal(10, 3, 100)
z1_ = np.random.normal(30, 6, 100)

x2_ = np.random.normal(8, 5, 100)
y2_ = np.random.normal(12, 3, 100)
z2_ = np.random.normal(10, 6, 100)

x3_ = np.random.normal(10, 3, 100)
y3_ = np.random.normal(3, 4, 100)
z3_ = np.random.normal(0, 3, 100)

x4_ = np.random.normal(5, 2, 100)
y4_ = np.random.normal(-10, 5, 100)
z4_ = np.random.normal(20, 6, 100)

x5_ = np.random.normal(-10, 3, 100)
y5_ = np.random.normal(20, 5, 100)
z5_ = np.random.normal(15, 5, 100)

x6_ = np.random.normal(-5, 3, 100)
y6_ = np.random.normal(-5, 5, 100)
z6_ = np.random.normal(6, 5, 100)

k1 = np.array((x1_, y1_, z1_))
data1 = k1.T
k2 = np.array((x2_, y2_, z2_))
data2 = k2.T
k3 = np.array((x3_, y3_, z3_))
data3 = k3.T
k4 = np.array((x4_, y4_, z4_))
data4 = k4.T
k5 = np.array((x5_, y5_, z5_))
data5 = k5.T
k6 = np.array((x6_, y6_, z6_))
data6 = k6.T

cnt_list = [0, 0, 0, 0, 0, 0]
sum = 0
for i in range(100):
    list = []
    list.append([distance.euclidean(data1[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data1[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data1[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data1[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data1[i], cluster_centers_k[4]), 4])
    list.sort()
    if list[0][0] > min_list[list[0][1]]:
        sum+=list[0][0]
        cnt_list[5]+=1
        continue
    cnt_list[list[0][1]]+=1
    sum+=list[0][0]
    #print(list[0], list[4])
print("data1: ",cnt_list, sum/100)

cnt_list = [0, 0, 0, 0, 0, 0]
sum = 0
for i in range(100):
    list = []
    list.append([distance.euclidean(data2[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data2[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data2[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data2[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data2[i], cluster_centers_k[4]), 4])
    list.sort()
    if list[0][0] > min_list[list[0][1]]:
        sum+=list[0][0]
        cnt_list[5]+=1
        continue
    cnt_list[list[0][1]]+=1
    sum+=list[0][0]
    #print(list[0], list[4])
print("data2: ",cnt_list, sum/100)

cnt_list = [0, 0, 0, 0, 0, 0]
sum = 0
for i in range(100):
    list = []
    list.append([distance.euclidean(data3[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data3[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data3[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data3[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data3[i], cluster_centers_k[4]), 4])
    list.sort()
    if list[0][0] > min_list[list[0][1]]:
        sum+=list[0][0]
        cnt_list[5]+=1
        continue
    cnt_list[list[0][1]]+=1
    sum+=list[0][0]
    #print(list[0], list[4])
print("data3: ",cnt_list, sum/100)

cnt_list = [0, 0, 0, 0, 0, 0]
sum = 0
for i in range(100):
    list = []
    list.append([distance.euclidean(data4[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data4[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data4[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data4[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data4[i], cluster_centers_k[4]), 4])
    list.sort()
    if list[0][0] > min_list[list[0][1]]:
        sum+=list[0][0]
        cnt_list[5]+=1
        continue
    cnt_list[list[0][1]]+=1
    sum+=list[0][0]
    #print(list[0], list[4])
print("data4: ",cnt_list, sum/100)

cnt_list = [0, 0, 0, 0, 0, 0]
sum = 0
for i in range(100):
    list = []
    list.append([distance.euclidean(data5[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data5[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data5[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data5[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data5[i], cluster_centers_k[4]), 4])
    list.sort()
    if list[0][0] > min_list[list[0][1]]:
        sum+=list[0][0]
        cnt_list[5]+=1
        continue
    cnt_list[list[0][1]]+=1
    sum+=list[0][0]
    #print(list[0], list[4])
print("data5: ",cnt_list, sum/100)

cnt_list = [0, 0, 0, 0, 0, 0]
sum = 0
for i in range(100):
    list = []
    list.append([distance.euclidean(data6[i], cluster_centers_k[0]), 0])
    list.append([distance.euclidean(data6[i], cluster_centers_k[1]), 1])
    list.append([distance.euclidean(data6[i], cluster_centers_k[2]), 2])
    list.append([distance.euclidean(data6[i], cluster_centers_k[3]), 3])
    list.append([distance.euclidean(data6[i], cluster_centers_k[4]), 4])
    list.sort()
    if list[0][0] > min_list[list[0][1]]:
        sum+=list[0][0]
        cnt_list[5]+=1
        continue
    cnt_list[list[0][1]]+=1
    sum+=list[0][0]
    #print(list[0], list[4])
print("data6: ",cnt_list, sum/100)



fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(131, projection='3d')
#ax.scatter(x1, y1, z1, c='green', marker='.', s=15, cmap='Greens')
#ax.scatter(x2, y2, z2, c='blue', marker='.', s=15, cmap='Greens')
#ax.scatter(x3, y3, z3, c='red', marker='.', s=15, cmap='Greens')
#ax.scatter(x4, y4, z4, c='black', marker='.', s=15, cmap='Greens')
#ax.scatter(x5, y5, z5, c='yellow', marker='.', s=15, cmap='Greens')
ax.scatter(X, Y, Z, c='grey', marker='.', s=15, cmap='Greens')

ax3 = fig.add_subplot(132, projection='3d')
ax3.scatter(x1, y1, z1, c='green', marker='.', s=15, cmap='Greens')
ax3.scatter(x2, y2, z2, c='blue', marker='.', s=15, cmap='Greens')
ax3.scatter(x3, y3, z3, c='red', marker='.', s=15, cmap='Greens')
ax3.scatter(x4, y4, z4, c='black', marker='.', s=15, cmap='Greens')
ax3.scatter(x5, y5, z5, c='yellow', marker='.', s=15, cmap='Greens')

ax2 = fig.add_subplot(133, projection='3d')
#ax2.scatter(X, Y, Z, c='grey', marker='.', s=15, cmap='Greens')
ax2.scatter(x, y, z, c=labels,marker='o',s=15, alpha=0.3)
ax2.scatter(cluster_centers_x, cluster_centers_y, cluster_centers_z,c='red', marker='x',s=100, alpha=1)

plt.show()