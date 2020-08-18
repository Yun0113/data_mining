# -*- coding: utf-8 -*-

#1.使用Kmeans聚类,设定k值为3,输出所有样本点的聚类结果,3个类别的类均值
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df=pd.DataFrame(datasets.load_iris()['data'],columns=datasets.load_iris()['feature_names']) 
iris = datasets.load_iris()
# 查看前数据，
df = iris.data
### 对数据集使用K-Means进行聚类
km = KMeans(n_clusters = 3)
km.fit(df)
# 打印聚类后3个簇的类均值
centers = km.cluster_centers_
print(centers)

#2.以花萼长度(sepal length),花瓣长度(petal length)为x,y轴,可视化聚类结果

from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

iris = datasets.load_iris()

### 对数据集进行探索
# 查看前两列数据，即花萼的长度和宽度
X = iris.data[:, (0,2)]
y = iris.target

# 将数据集中所有数据进行二维可视化展示
plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')
plt.xlabel('Spea1 Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)