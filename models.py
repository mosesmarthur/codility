import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
from tenacity import RetryAction
import cnn 
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn import metrics


data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')
x_train = data_train.iloc[:,0:98]
y_train = data_train['price']

x_test = data_test.iloc[:,0:98]
y_test = data_test['price']


def visualize(d1,d2,d3):
    model = ExtraTreesClassifier()
    model.fit(d1,d2)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=d1.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    feat_importances = feat_importances.nlargest(10)
    print(d1[feat_importances.keys()])
    print(d3[feat_importances.keys()])

    #plt.show()
    return d1[feat_importances.keys()],d3[feat_importances.keys()]

def pca(d1):
    ndata_train = StandardScaler().fit_transform(d1) # normalizing the features
    ndata_train = pd.DataFrame(ndata_train)
    pca_data= PCA(n_components=35)
    pca_data_train = pca_data.fit_transform(ndata_train)
    pca_data_train = pd.DataFrame(pca_data_train)
    print(pca_data_train)
    return pca_data_train

def linear(d1,d2,d3,d4):
    d1 = pd.DataFrame(d1)
    d2 = pd.DataFrame(d2)
    #Linear Regression 
    clf = linear_model.LinearRegression()
    clf.fit(d1, d2)
    result = {}
    result['score'] = clf.score(d1, d2)
    result['Coeff'] = clf.coef_  
    
    y_pred = clf.predict(d3)
    mae = metrics.mean_absolute_error(d4, y_pred)
    mse = metrics.mean_squared_error(d4, y_pred)
    r2 = metrics.r2_score(d4, y_pred)

    print("The Linear model performance for testing set")
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2))

    return result

def ridge(d1,d2,d3,d4):
    #Ridge CV
    clf = linear_model.RidgeCV()
    clf.fit(d1,d2)
    score = clf.score(d1, d2)
    result = {}
    result['score'] = clf.score(d1, d2)
    result['Coeff'] = clf.coef_  
    print(result)

    y_pred = clf.predict(d3)
    mae = metrics.mean_absolute_error(d4, y_pred)
    mse = metrics.mean_squared_error(d4, y_pred)
    r2 = metrics.r2_score(d4, y_pred)

    print("The RidgeCV performance for testing set")
    print("--------------------------------------")
    print('MAE is {}'.format(mae))
    print('MSE is {}'.format(mse))
    print('R2 score is {}'.format(r2))

    return result
    
def main():
    
    rdata_train, rdata_test = visualize(x_train,y_train,x_test)
    rdata_train = pd.DataFrame(rdata_train)
    rdata_test = pd.DataFrame(rdata_test)

    #rdata_test = visualize(x_test,y_test)
    #npcadata_train = pca(data_train)
    #correlationmatrix(npcadata_train)
    #linear_result = linear(rdata_train,y_train,rdata_test,y_test)
    ridge_result = ridge(rdata_train,y_train,rdata_test,y_test)
    #cnn_result = cnn.cnn_model(x_train,y_train,x_test,y_test)
    
if __name__ == "__main__":
    main()


    
