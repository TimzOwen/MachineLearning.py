from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

plt.style.use('ggplot')

#load dataset
iris = dataset.load_iris()

type(iris)

print(iris.keys())
