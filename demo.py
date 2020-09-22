import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



st.title(" Data set example ")

st.write("""
# Explore diffrent classifier and decide which is best?
 """)

dataset_name = st.sidebar.selectbox('select data set',('Iris','Breast Cancer','Wine')
)

st.write(f"## {dataset_name}Dataser")

classifier_name = st.sidebar.selectbox("select classifier",('KNN','SVM','Random Forest')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()

    x = data.data
    y = data.target
    return x,y


#get list of data set and display shape and total count
x,y = get_dataset(dataset_name)
st.write('Shape of dataset:',x.shape)
st.write('number of classes:',len(np.unique(y)))

#function to fatch the classifier  and display parametrs 

def add_perameters_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K',1,15)
        params['K']=K
    else:
        max = st.sidebar.slider('max',1,100)
        params['max']= max
        n_estimator = st.sidebar.slider('n_estimator',1,100)
        params['n_estimator']=n_estimator
    return params

# based on classifer name display of attributes of classifer
params = add_perameters_ui(classifier_name)


def get_classifier(clf_name,params):
    clf = None
    if clf_name == 'SVM':
        clf= SVC(C=params['C'])
    elif clf_name== 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimator'],max_depth=params['max'],random_state=1234)
    return clf

clf = get_classifier(classifier_name,params)

##### CLASSIFICATION #####

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test,y_pred)

st.write(f'classifier = {classifier_name}')
st.write(f'Accuracy=',acc)

######plot data set###########
pca = PCA(2)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar()
#plt.show()
st.pyplot(fig)








