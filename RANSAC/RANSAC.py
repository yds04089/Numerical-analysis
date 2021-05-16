import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial import distance

a=2
b=-1

lin=[-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
x=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
y_=a*x+b

noise=math.sqrt(2)*np.random.randn(12)

yy=y_+noise

print('x', x)
print('y', yy)


y=yy

x_bar=x.mean()
y_bar=y.mean()

w_12=((x-x_bar)*(y-y_bar)).sum()/((x-x_bar)**2).sum()
b_12=y_bar-w_12*x_bar

y_12 = w_12*x+b_12

print('w_12: {:.2f}'.format(w_12))
print('b_12: {:.2f}'.format(b_12))



criterion=0.03
w_6=10
b_6=10
before_in=0
cnt=0
x_6=[]
y_6=[]
in_cnt=0
dis = 100
before_dis = 100

while True:
   tmp=random.sample(lin,6)
   tmp_x=[]
   tmp_y=[]
   in_cnt = 0
   for i in range(6): 
      tmp_x.append(x[tmp[i]+5])
      tmp_y.append(y[tmp[i]+5])
   sample_x=np.array(tmp_x)
   sample_y=np.array(tmp_y)

   x_bar=sample_x.mean()
   y_bar=sample_y.mean()

   w=((sample_x-x_bar)*(sample_y-y_bar)).sum()/((sample_x-x_bar)**2).sum()
   b=y_bar-w*x_bar

   for i in range (12):
      exp_y = w*x[i]+b
      dif = abs(y[i] - exp_y)
      if (dif < 1): in_cnt+=1

   tmp_y_ = w*np.float64(x) + b
   dis = distance.euclidean(y_, tmp_y_)


   #print(error)
   if before_in<in_cnt:
      w_6=w
      b_6=b
      before_in=in_cnt
      x_6 = tmp_x
      y_6 = tmp_y
   elif before_in == in_cnt and before_dis > dis:
      w_6=w
      b_6=b
      before_in=in_cnt
      x_6 = tmp_x
      y_6 = tmp_y

   cnt+=1

   if in_cnt>8 or cnt>10000:
      break

print(w_6, b_6)
y_6_final = w_6*np.float64(x) + b_6
print('w_6: {:.2f}'.format(w_6))
print('b_6: {:.2f}'.format(b_6))
print(in_cnt,cnt)
print("6 거리: ", distance.euclidean(y_, y_6_final))
print("12 거리: ", distance.euclidean(y_, y_12))
print("6 - 12 거리: ", distance.euclidean(y_6_final, y_12))
plt.figure(figsize=(10, 7))
plt.plot(x, y_, color='g', label='y = 2*x -1')
plt.plot(x, y_12, color='b', label='by 12')
plt.plot(x, y_6_final, color='r', label='by RANSAC')

plt.scatter(x, yy, label='data')
plt.scatter(x_6, y_6, color='r', label='data_RANSAC')
plt.legend(fontsize=12)
plt.show()

