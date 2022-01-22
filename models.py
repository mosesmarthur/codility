import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
from sklearn import linear_model
import cnn 
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

def pca(data_train):
    ndata_train = StandardScaler().fit_transform(data_train) # normalizing the features
    ndata_train = pd.DataFrame(ndata_train)
    pca_data= PCA(n_components=25)
    pca_data_train = pca_data.fit_transform(ndata_train)
    pca_data_train = pd.DataFrame(pca_data_train)
    return pca_data_train

def correlationmatrix(data_train):
    data_train_cor = data_train.corr() # Generate correlation matrix
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = data_train_cor.columns,
            y = data_train_cor.index,
            z = np.array(data_train_cor)
        )
    )
    fig.show()

def linear(x_train,y_train):
    #Linear Regression 
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    result = {}
    result['score'] = reg.score(x_train, y_train)
    result['Coeff'] = reg.coef_    
    return(result)

def ridge(x_train,y_train):
    #Ridge CV
    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
    reg.fit(x_train,y_train)
    score = reg.score(x_train, y_train)
    result = {}
    result['score'] = reg.score(x_train, y_train)
    result['Coeff'] = reg.coef_    
    return(result)
    
def main():
    
    x_train = data_train.iloc[:,0:98]
    y_train = data_train['price']

    x_test = data_test.iloc[:,0:98]
    y_test = data_test['price']

    npcadata_train = pca(data_train)
    correlationmatrix(npcadata_train)
    #linear_result = linear(x_train,y_train)
    #ridge_result = ridge(x_train,y_train)
    #cnn_result = cnn.cnn_model(x_train,y_train)
    
if __name__ == "__main__":
    main()


    
