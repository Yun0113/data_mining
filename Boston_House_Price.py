# -*- coding: utf-8 -*-

#1.画出RM,DIS,PTRATIO,LSTAT与y的散点图,分析特征与y是否有线性关系?

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

x_df = pd.DataFrame(datasets.load_boston()['data'],columns=datasets.load_boston()['feature_names']) #X
y_df = pd.DataFrame(datasets.load_boston()['target'],columns=['y']) #Y
df = x_df.join(y_df)
RM = df.RM
DIS = df.DIS
PTRATIO = df.PTRATIO
LSTAT = df.LSTAT
y = df.y
# 将数据集中所有数据的散点图
plt.figure()
plt.subplot(221) 
#RM与y的散点图
z1 = np.polyfit(RM, y,1) # 拟合
p1 = np.poly1d(z1)
#print("RM与y的预测回归方程:",p1) # 在屏幕上打印拟合曲线
yvals=p1(RM) # 也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(RM, yvals, 'r',label='polyfit values')
plt.scatter(RM, y,c='g',s=10.)
plt.xlabel('RM')
plt.ylabel('y')

plt.show()
#DIS与y的散点图
plt.subplot(222)
z2 = np.polyfit(DIS, y,1) # 拟合
p2 = np.poly1d(z2)
#print("DIS与y的预测回归方程:",p2) # 在屏幕上打印拟合曲线
yvals=p2(DIS) # 也可以使用yvals=np.polyval(z1,x)
plot2=plt.plot(DIS, yvals, 'r',label='polyfit values')
plt.scatter(DIS, y,c='b',s=10.)
plt.xlabel('DIS')
plt.ylabel('y')

plt.show()
#PTRATIO与y的散点图
plt.subplot(223)
z3 = np.polyfit(PTRATIO, y,1) # 拟合
p3 = np.poly1d(z3)
#print("PTRATIO与y的预测回归方程:",p3) # 在屏幕上打印拟合曲线
yvals=p3(PTRATIO) # 也可以使用yvals=np.polyval(z1,x)
plot3=plt.plot(PTRATIO, yvals, 'b',label='polyfit values')
plt.scatter(PTRATIO, y,c='r',s=10.)
plt.xlabel('PTRATIO')
plt.ylabel('y')

plt.show()
#LSTAT与y的散点图
plt.subplot(224)
z4 = np.polyfit(LSTAT, y,1) # 拟合
p4 = np.poly1d(z4)
#print("LSTAT与y的预测回归方程:",p4) # 在屏幕上打印拟合曲线
yvals=p4(LSTAT) # 也可以使用yvals=np.polyval(z1,x)
plot4=plt.plot(LSTAT, yvals, 'g',label='polyfit values')
plt.scatter(LSTAT, y,c='y',s=10.)
plt.xlabel('LSTAT')
plt.ylabel('y')
plt.show()

#2.尝试进行线性回归,使用RM,DIS,PTRATIO,LSTAT预测房价y,写出回归方程
x_df = pd.DataFrame(datasets.load_boston()['data'],columns=datasets.load_boston()['feature_names']) #X
y_df = pd.DataFrame(datasets.load_boston()['target'],columns=['y']) #Y
df = x_df.join(y_df)
X = x_df[["RM","DIS","PTRATIO","LSTAT"]]#选取指标
y = y_df
lr = LinearRegression()#线性回归
lr.fit(X,y)
print('coefficients(b1,b2...):',lr.coef_)#回归系数

#所以回归方程为： y = 4.2238*RM-0.5519*DIS-0.9736*PTRATIO-0.6654*LSTAT


#3.解释下RM与Y的关系?
'''首先从整体回归y = 4.2238*RM-0.5519*DIS-0.9736*PTRATIO-0.6654*LSTAT来看，RM与y存在正相关性，
同时RM与y的预测回归方程:  9.102 x - 34.67，则可以看出RM与Y存在正相关关系'''

#4对某新小区,其RM=8,DIS=2,PTRATIO=12,LSTAT=22,预测该小区房价
list = [[8,2,12,22]]
x_test = pd.DataFrame(list)
y_pred = lr.predict(x_test)
print("小区房价预测:",y_pred)




