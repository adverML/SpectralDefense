#!/usr/bin/env python3
"""
https://towardsdatascience.com/anomaly-detection-with-isolation-forest-visualization-23cd75c281e2
https://machinelearningmastery.com/anomaly-detection-with-isolation-forest-and-kernel-density-estimation/
"""
import pdb
import numpy as np
from sklearn.ensemble import IsolationForest



def IF(args, logger, X_train, y_train, X_test, y_test):

    n_estimators = 300
    # max_samples = 'auto'
    max_samples = X_train.shape[0]
    warm_start = False
    contamination = 'auto'
    # contamination=.03

    clf = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, n_jobs=-1, random_state=21, warm_start=warm_start, verbose=0) 
    logger.log(clf)

    clf.fit(X_train, y_train)
    # test_score = clf.score(X_test, y_test)
    # print("test score: ", test_score)

    y_hat = clf.predict(X_test, y_test)
    dec_f = clf.decision_function(X_test)
    score = clf.score_samples(X_test)

    
    outlier_index = np.where(y_hat==-1)
    # y_hat_pr = clf.predict_proba(X_test)[:, 1]

    print('outlier_index', outlier_index)

    pdb.set_trace()

    y_hat_pr = y_hat.copy()
    y_hat_pr = y_hat[outlier_index]

    return clf, y_hat, y_hat
