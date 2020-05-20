from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

plt.style.use('ggplot')

#load dataset
iris = dataset.load_iris()

type(iris)

print(iris.keys())


#check the type of the data
type(iris.data), type(iris.target) #shows they are numpy arrays

irish.data.shape #shows (150,4) rows and columns respectively

#print array index
iris.target_names

#EXPLORATORY DATA ANALYSIS (EDA)

#assign to values
x = iris.data
y = iris.target

df = pd.DataFrame(x, columns=iris.feature_names)

print(df.head())

#visual EDA
_ = pd.plotting.scatter_matrix(df,c=y,figsize=[8,s=150,marker=D])


#CLASSIFYING USING K-NEAREST ALGORITHM
#using sklearn to fit a classifier
from sklearn.neighbors import KNeighborsClassifier

#instantiate the classifier
knn = KNeighborsClassifier(n_neighbors=6)

#fit the data of iris into the model
knn.fit(iris['data'],iris['target'])

#print shape
iris['data'].shape  #(150,4)
iris['target'].shape

#predicting on unlabeled data
x_new = np.array([[5.6,2.8,5,0.2],
                  [4,2.5,6,8.6],
                  [1.8,2.6,5,10.5]])

#now perform prediction
prediction = knn.predict(x_new)

#check shape
x_new.shape

#print prediction
print('prediction: {}'.format(prediction))

[1,1,0] #output prediction
