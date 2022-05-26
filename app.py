import streamlit as st 
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

#streamlit hello -> see magic over http://localhost:8501/
st.title('Streamlit Example')

st.write(""" 
# EXPLORE DIFFERENT CLASSIFIERS
Normal text
""")

dataset_name = st.sidebar.selectbox('Select Built-In Dataset: ',('Iris','Breast Cancer','Wine Dataset'))
# st.sidebar.write(dataset_name)

classifier_name = st.sidebar.selectbox('Select Classifier: ',('KNN','SVM','Random Forest'))

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data  
    y = data.target  
    return x,y

x,y=get_dataset(dataset_name)
st.write('Shape of the Datasets: ',x.shape)
st.write('Number of classes: ',len(np.unique(y)))


def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'KNN':
        K = st.sidebar.slider('K',1,15)
        params['K'] = K
    elif classifier_name == "SVM":
        C =  st.sidebar.slider('K',0.1,10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth',2,15) #no of depth for each tree 
        n_estimators = st.sidebar.slider('n_estimators',1,100)   
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name )        

def get_classifier(classfier_name,params):
    if classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == "SVM":
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
        max_depth = params['max_depth'],random_state=1234)
    return clf
clf = get_classifier(classifier_name,params)

# Classification

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2, random_state=1234)

clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)


acc = accuracy_score(Y_test, Y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")


#PLOT

# Multi-dimension array into 2D

pca = PCA(2)
X_projected = pca.fit_transform(x) 

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.xlabel('Principal Component 2')
plt.colorbar()

#plt.show()
st.pyplot(fig)