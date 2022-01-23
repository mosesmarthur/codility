import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
import cnn 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')
x_train = data_train.iloc[:,0:98]
y_train = data_train['price']

x_test = data_test.iloc[:,0:98]
y_test = data_test['price']


def visualize(d1,d2):
    model = ExtraTreesClassifier()
    model.fit(d1,d2)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=d1.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    feat_importances = feat_importances.nlargest(10)    
    #plt.show()
    return d1[feat_importances.keys()]

def pca(d1):
    ndata_train = StandardScaler().fit_transform(d1) # normalizing the features
    ndata_train = pd.DataFrame(ndata_train)
    pca_data= PCA(n_components=35)
    pca_data_train = pca_data.fit_transform(ndata_train)
    pca_data_train = pd.DataFrame(pca_data_train)
    print(pca_data_train)
    return pca_data_train

def linear(x_train,y_train):
    #Linear Regression 
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    result = {}
    result['score'] = reg.score(x_train, y_train)
    result['Coeff'] = reg.coef_  
    print('This is Linear',result)  
    return result

def ridge(x_train,y_train):
    #Ridge CV
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit(x_train,y_train)
    score = reg.score(x_train, y_train)
    result = {}
    result['score'] = reg.score(x_train, y_train)
    result['Coeff'] = reg.coef_    
    return result
    
def main():
    
    #rdata_train = visualize(x_train,y_train)
    #rdata_train = pd.DataFrame(rdata_train)
    #npcadata_train = pca(data_train)
    #correlationmatrix(npcadata_train)
    #linear_result = linear(rdata_train,y_train)
    #ridge_result = ridge(x_train,y_train)
    cnn_result = cnn.cnn_model(x_train,y_train,x_test,y_test)
    
if __name__ == "__main__":
    main()


    
