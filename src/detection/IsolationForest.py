#!/usr/bin/env python3
    
from sklearn.ensemble import IsolationForest

def IF(args, logger, X_train, y_train, X_test, y_test):

    # (n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=random_state)
    clf = IsolationForest(n_estimators=100,max_samples='auto',contamination='auto', n_jobs=-1, random_state=None, verbose=0) 
    logger.log(clf)

    clf.fit(X_train, y_train)
    # test_score = clf.score(X_test, y_test)
    # print("test score: ", test_score)

    return clf