import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

#%%
# let us import the data 

iris = sns.load_dataset('iris')
iris.info

"""Three classes in the Iris dataset:

    Iris-setosa (n=50)
    Iris-versicolor (n=50)
    Iris-virginica (n=50)
    
The four features of the Iris dataset:

sepal length in cm
sepal width in cm
petal length in cm
petal width in cm  """ 

iris.columns

#%% data visualization

sns.pairplot(data = iris, hue = 'species' )

setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)

#%% Split the data and train the model 

from sklearn.model_selection import train_test_split

X = iris.drop(['species'], axis = 1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#%% Using GridSearch to improve the parameters 

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1,10,100,1000], 'gamma':[1, 0.1, 0.01, 0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=5)


grid.fit(X_train,y_train)


grid.best_params_

grid_predction = grid.predict(X_test)
print(classification_report(y_test, grid_predction))







