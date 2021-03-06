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


#Prediction002 
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party',axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)
X_new = knn.predict(X_new)

# Predict and print the label for the new data point X_new
new_prediction = np.array(X_new)
print("Prediction: {}".format(new_prediction))


#UPNEXT- MEASURING MODEL PERFORMANCE
#split your data into training and testing

#train/test split
from sklearn.model_selection import train_test_split

X_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

#instantiate the classifier
knn = KNeighborsClassifier(n_neighbors=8)

#feed in the test data
knn.fit(X_train, y_train)

#predict
y_pred = knn.predict(x_test)

print("test set prediction:\n {}".format(y_pred))

#check accuracy test in percentage
knn.score(x_test, y_test)   #0.95555555 output sample meaning 95% accurate

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#PERform MNIST no. detection
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


#REGRESSION WITH BOSTON HOUSE DATASET

#first import and load dataset
boston = pd.load_csv('boston.csv')

#display the recent data
print(boston.head())

#now create feature and Target
X = boston.drop('MEDV',axis=1).Values
y = boston['MEDV'].values

#predict house from a single feature column no. 5
X_room = X[:,5]

#you can check the type of the data
type(X_rooms), type(y) #numpy arrays

#reshape
y = y.reshape(-1 ,1)
X_rooms = X_rooms.reshape(-1, 1)

#plot house numbers vs number of rooms
plt.scatter(X_rooms, y)
plt.ylabel('value of houses/1000($)')
plt.xlabel('No. of rooms')
plt.show();

#Fitting a Linear Regression Model
import numpy as np
from sklearn.linear_model import LinearRegression

#instantiate the model
reg = LinearRegression()

#fit the model
reg.fit(X_rooms, y)

#prediction_space
prediction_space =  np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)

#plot the line
plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()


#Importing data for supervised Learning to predict Life expectancy of a country using their GDP
# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

#output
    Dimensions of y before reshaping: (139,)
    Dimensions of X before reshaping: (139,)
    Dimensions of y after reshaping: (139, 1)
    Dimensions of X after reshaping: (139, 1)
    


#THE BASICS OF LINEAR REGRESSION

#Linear Regression on all features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg_all = LinearRegression()

reg_all.fit(X_train, y_train)
y_pred=reg_all.predict(X_test)
reg_all.score(X_test, y_test)



#LINEAR REGRESSION TO DO LIFE EXPECTANCY PREDICTION
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility,y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


#TRAIN/TEST FOR LINEAR REGRESSION
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


#CROSS VALIDATION
#spliting the data into test and train and using Folder mathods for training.
#Hold one folder for testing and iterate through the rest as training data

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

cv_results = cross_val_score(reg, X, y, cv=5)

print(cv_results) #prints an array

#you can also compute the mean 
np.mean(cv_results)

#EX 002
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

#outputs 
[0.81720569 0.82917058 0.90214134 0.80633989 0.94495637]
Average 5-Fold CV Score: 0.8599627722793232


#K-FOLDER CV_COMPARISON
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X,y, cv=10)
print(np.mean(cvscores_10))

#output
0.8718712782622108
0.8436128620131201

#using the time loops to check for comparison,
In [3]: %timeit cross_val_score(reg, X, y, cv = 3)
100 loops, best of 3: 7.83 ms per loop

In [10]: %timeit cross_val_score(reg, X, y, cv = 10)
10 loops, best of 3: 24.5 ms per loop


  
#REGULARIZED REGRESSION
# 1.0 ---> Ridge regression
from sklearn.linear_model import Ridge
X_train, x_test,y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

ridge = Ridge(alpha=1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

# 2.0 ----> Lasso Regresion
from sklearn.linear_model import Lasso
X_train, x_test,y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

lasso = Ridge(alpha=1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = ridge.predict(X_test)
lasso.score(X_test, y_test)

#LASSO for feature selection
from sklearn.linear_model import Lasso

names = boston.drop('MEDV',axis=1).columns

lasso = Lasso(alpha=0.1)

lasso_coef = lasso.fit(X,y).coef_
 
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso_coef = lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso_coef.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()



#REGULARIZATION II (RIDGE REGRESSION)
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


#CONFUSION MATRIX
#used to improve model performance

#import confusion matrix and classification report
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#instatiate a classifier
knn = KNeighborsClassifier(n_neighbors=8)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#confusion metrix
print(confusion_matrix(y_test, y_pred))

#print all relevant matrix
print(classification_report(y_test, y_pred))

#EX 002
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40,
random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#output
     precision    recall  f1-score   support
    
              0       0.77      0.85      0.81       206
              1       0.62      0.49      0.55       102
    
    avg / total       0.72      0.73      0.72       308

    #LOGISTIC REGRESSION
#uses a decision boundary to make decicsion of 0.5>1 0.4<0
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40,
random_state=42)

logreg.fit(X_train, y_train)

y_predict = logreg.predict(X_test)

 
 #Plotting ROC curve variance
 from sklearn.metrics import roc_curve 
 
 y_pred_prob = logreg.predict_proba(X_test)[:,1]
 
 fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
 
 #plot the results
 plt.plot([0,1], [0,1], 'k--')
 plt.plot(fpr, tpr, label='Logistic Regression ')
 plt.xlable('False positive Rate')
 plt.ylabel('True positive Rate')
 plt.title(Logistic Regression ROC Curve)
 plt.show()


# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#output 
             precision    recall  f1-score   support

          0       0.83      0.85      0.84       206
          1       0.69      0.66      0.67       102

avg / total       0.79      0.79      0.79       308

#now plot a ROC curve
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
 

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression ')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#USING AUC TO FIND THE AREA
from sklearn.metrics import roc_auc_score

logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

logreg.fit(X_train, y_train)

y_pred_prob = logreg.predict(X_test)[:, 1]

roc_auc_score(y_test,y_pred_prob)

#output 
0.989856525456


#USING AUC CROSS-VALIDATION
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(logreg, X,y, cv=5, scoring='roc_auc')

print(cv_scores)


#HYPER PARAMETER TUNNING 
#using Grid search score
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': np.arange(1, 50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, param_grid, cv=5)

knn_cv.fit(X,y)
knn_cv.best_params_
knn_cv.best_score_


#Decision tree
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


#HOLD-OUT DATASETS
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


#Hold-out set in pratice 2

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV,train_test_split
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))




#PREPROCESSING DATA
#encoding dummy variables
import pandas as pd

#read the csv
df = pd.read_csv('auto.csv')

df_origin = pd.get_dummies(df)

print(df-origin.head())

#drop asia
df_origin = df_origin.drop('origin_Asia', axis=1)

print(df_origin.head())


#EX 2
# Import pandas
import pandas as pd 

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


#EX 3 dummies
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = df_region.drop('Region_America', axis=1)

# Print the new columns of df_region
print(df_region.columns)

#now run throught the whole model
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)

#output
 [0.86808336 0.80623545 0.84004203 0.7754344  0.87503712]

  
  
  HANDLING MISSING DATA 
#using PIMA Indians datasets,

df = pd.load_csv('diabetes.csv')
df.info() #shows the data summary
#show latest record
print(df.head())

#drop missing data 
df.insulin.replace(0, np.nan, inplace=True)
df.tricep.replace(0, np.nan, inplace=True)
df.bmi.replace(0, np.nan, inplace=True)
df.info() #to show table info

#drop all rows containing missing data 
df = df.dropna()
df.shape

#IMPUTING MISSING DATA
from sklearn.preprocessing import Imputer 
imp = Imputer(missing_values='NAN', strategy='mean',axis=0)
imp.fit(X)
X=imp.transform(X)

#IMPUTING WITH PIPELINES
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import imputer 

imp = Imputer(missing_value='NaN', strategy='mean',axis=0)

logreg = LogisticRegression()

steps = [('imputation', imp),
         'logistic_regressioin', logreg]

Pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fit the pipeline
pipeline.fit(X_train, y_train)

y_pred = Pipeline.predict(X_test)

#compute accuracy
pipeline.score(X_test, y_test)


#IMPUTING MISSING DATA WITH PIPELINE    
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

#ML PIPELINE002
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(pipeline.score(X_test, y_test))

#output
 0.9694656488549618


# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


                precision    recall  f1-score   support
    
       democrat       0.99      0.96      0.98        85
     republican       0.94      0.98      0.96        46
    
    avg / total       0.97      0.97      0.97       131
    
    
    

#CENTERING AND SCALING DATA ON MACHINES
from sklearn.preprocessing import scale

X_scaled = scale(X)

np.mean(X), np.std(X)

np.mean(X_scaled), np.std(X_scaled)

#SCALING IN PIPELINE

from sklearn.preprocessing import StandardScaler

steps = [('scaler',StandardScaler()),
         ('Knn',KNeighborsClassifier())]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=22)

knn_scaled = pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy_score(y_test, y_pred)

Knn_unscaled  = KNeighborsClassifier().fit(X_train, y_train)

Knn_unscaled.score(X_test, y_test)


#cross validation CV and Scaling in pipeline
steps = [('scaler',StandardScaler()),
         ('Knn',KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameters = {Knn__n_neighbors: np.arange(1,50)}

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=21)

cv = GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

#print best param
print(cv.best_params_)

#print_score
print(cv.score(X_test, y_test))

#print classfication report
print(classification_report(y_test, y_pred))


#Pipelining Recap
# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))




# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

    Accuracy with Scaling: 0.7700680272108843
    Accuracy without Scaling: 0.6979591836734694
    
    
    
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

    Accuracy: 0.7795918367346939
                 precision    recall  f1-score   support
    
          False       0.83      0.85      0.84       662
           True       0.67      0.63      0.65       318
    
    avg / total       0.78      0.78      0.78       980
    
    Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}
    
      Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
    Tuned ElasticNet R squared: 0.8862016570888217
