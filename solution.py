from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
    


def solution(data_train,data_test)
    X_train = data_train.iloc[:,0:98]
    y_train = data_train.price
    X_cv = data_test.iloc[:,0:98]
    y_cv = data_test.price



    from sklearn.linear_model import RidgeCV
    clf1 = RidgeCV(alphas=4, normalize=True)
    clf1.fit(X_train,y_train)
    clf1.score(X_train,y_train)
    pred1 = clf1.predict(X_cv)



    clf2 = LassoCV(alphas=4, normalize=True)
    clf2.fit(X_train,y_train)
    clf2.score(X_train,y_train)
    pred2 = clf1.predict(X_cv)
    pred2
    clf2.coef_


    clf3 = ElasticNet(alpha=1, l1_ratio=0.5, normalize=True)
    clf3.fit(x_train,y_train)
    pred3 = clf3.predict(X_cv)
    pred3
    clf3.coef_

    D = {'Ridge': {'alpha': alpha1, 'pred': pred1, 'coef_': clf1.coef_},
         'Lasso': {'alpha': alpha2, 'pred': pred2, 'coef_': clf2.coef_},
         'ElasticNet': {'alpha': alpha3,'pred' :pred3 , 'coef_': clf3.coef_}}
    return(D)
