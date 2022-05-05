#!/usr/bin/env python3

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

def LR(args, logger, X_train, y_train, X_test, y_test):

    clf = LogisticRegression(max_iter=args.num_iter)

    # penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
    # dual bool, default=False --> primal
    # tolf loat, default=1e-4
    # C float, default=1.0
    # intercept_scaling float, default=1
    # class_weightdict or ‘balanced’, default=None
    # random_stateint, RandomState instance, default=None
    # max_iter int, default=100
    # multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
    
    # scaler  = MinMaxScaler().fit(X_train)
    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)


    logger.log(clf)
    clf.fit(X_train, y_train)
    logger.log("train score: " + str(clf.score(X_train, y_train)) )
    logger.log("test score:  " + str(clf.score(X_test,  y_test)) )

    y_hat =    clf.predict(X_test)
    y_hat_pr = clf.predict_proba(X_test)[:, 1]

    return clf, y_hat, y_hat_pr