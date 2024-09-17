#Importing all the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn. linear_model import LinearRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('bottle.csv')
df_binary = df[['Salnty','T_degC']]

#Taking only the selected two attributes from the dataset
df_binary.columns = ['Sal','Temp']

#display the first 5 rows
df_binary.head()

sns.lmplot(x = "Sal", y = "Temp", data = df_binary, ci = None)
plt.show()

# Eliminating NaN or missing input numbers
df_binary.fillna(method = 'ffill', inplace = True)

X = np.array(df_binary['Sal']).reshape(-1, 1)
y = np.array(df_binary['Temp']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array
# since each dataframe contains only one column
df_binary.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color = 'b')
plt.plot(X_test, y_pred, color = 'k')

plt.show()

df_binary500 = df_binary[:][:500] 
    
# Selecting the 1st 500 rows of the data 
sns.lmplot(x ="Sal", y ="Temp", data = df_binary500,
                                order = 4, ci = None)

df_binary500.fillna(method ='ffill', inplace = True) 
  
X = np.array(df_binary500['Sal']).reshape(-1, 1) 
y = np.array(df_binary500['Temp']).reshape(-1, 1) 
  
df_binary500.dropna(inplace = True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
  
regr = LinearRegression() 
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='k') 
  
plt.show() 


# initialize list of lists
data = [[32,200,0,1], [28,143,0,1], [25,150,1,0],[29,220,1,0],[36,196,3,1]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Age', 'Weight', 'Injuries', 'Results'])

X = df[['Age','Weight','Injuries']]
y = df['Results']
features=['Age','Weight','Injuries']

## Fitting the model
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

## Ploting the tree
tree.plot_tree(dtree, feature_names=features)
plt.show()