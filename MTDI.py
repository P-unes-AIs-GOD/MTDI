import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

file = 'my-df.xlsx'
xl = pd.ExcelFile(file)
 
array1 = xl.parse('Sheet1')
   

sb.pairplot(array1) #Searching for a good corilation
plt.show()# Inch & Weight are result of search

array1.boxplot('Inch')# Searching for Bad Data of Inch fature property
plt.show()

array1.boxplot('Weight')# Searching for Bad Data of Weight fature property
plt.show()

x=np.array(array1[['Inch','Weight','Price','Ram']]) # separate properties and target col
y=np.array(array1.Target)

plt.scatter(x[:,0],x[:,1],c=y) #See the data from a better perspective
plt.xlabel('Inch')
plt.ylabel('Weight')
plt.show()

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y) # the chart of Testing and Training for choosing better n_neighbors
neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train, y_train)
    train_accuracy[i] = knn_model.score(x_train, y_train)
    test_accuracy[i] = knn_model.score(x_test, y_test)
    
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('number of Neighbors')
plt.ylabel('Accuracy')
plt.show() # int 4

knn =  KNeighborsClassifier(n_neighbors=4, metric='minkowski',p=2) # Now just Do it :)
knn.fit(x,y)








# 0 = Laptop
# 1 = Tablet
# 2 = Phone

#['Inch','Weight','Price','Ram']


x_new = np.array([[6.5,0.45,4500,3]])
y_new = knn.predict(x_new)
y_new





