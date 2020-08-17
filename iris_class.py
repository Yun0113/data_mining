# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X,y=datasets.load_iris(return_X_y=True) #X与y
target_names=datasets.load_iris().target_names #y的值列表:0:setosa,1:versicolor,2:virginica
feature_names=datasets.load_iris().feature_names #特征X的名称列表
#划分训练集和测试集
feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size=0.33, random_state=56)

dt_model = DecisionTreeClassifier(max_depth=3)#最大树深度为3
dt_model.fit(feature_train, target_train)
predict_results = dt_model.predict(feature_test)#测试集训练结果
scores = dt_model.score(feature_test, target_test)#得分   
print(predict_results)
print(target_test)
print(accuracy_score(predict_results, target_test))#训练结果准确度

#预测sepal length=6,sepal width=1,petal length=3,petal width=1最可能是什么类型的花
test = [6.,1.,3.,1.]
np.array(test)
predict_result = dt_model.predict(test)#测试结果
print(predict_result)#预测结果 结果为1，则最可能是‘versicolor’


