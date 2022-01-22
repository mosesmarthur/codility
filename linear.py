from unittest import result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
from sklearn import linear_model

data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')


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

    linear_result = linear(x_train,y_train)
    ridge_result = ridge(x_train,y_train)
    
if __name__ == "__main__":
    main()


    
