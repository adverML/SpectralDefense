#!/usr/bin/env python3


from sklearn.ensemble import RandomForestClassifier
import copy
import pdb

def RF(args, logger, X_train, y_train, X_test, y_test):
    
    # trees = [ 300 ]

    # max_depths = np.linspace(1, 32, 32, endpoint=True)
    # max_depths = [5, 10, 100, 500, 800, 1000]
    # max_depths = [5, 10, 100, 500, 800]
    # max_leaf_nodes = [5, 10, 50, 100, 500, 800, 1000]
    max_depths     = [None]
    max_leaf_nodes = [None]

    # trees = [100, 200, 300, 400, 500]
    # trees = [600, 700, 800, 900]
    # trees = [ 200, 300, 400, 500, 600, 800, 900, 1200 ]
    # trees = [25, 50, 75, 100, 300, 600]
    # trees = [ 25, 50, 100, 500, 1000, 2000 ]
    # trees = [ 300, 1000, 2000 ]
    trees = [ 300 ]



    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1) 
    save_clf = copy.deepcopy(clf)
    test_score_save = 0

    for max_leaf_node in max_leaf_nodes:
        for max_depth in max_depths:
            for tr in trees:
                clf.set_params(n_estimators=tr, max_depth=max_depth, max_leaf_nodes=max_leaf_node, random_state=21, verbose=0)
                clf.fit(X_train, y_train)

                test_score = clf.score(X_test, y_test)
                logger.log("max_leaf " + str(max_leaf_node) + " max_depth " + str(max_depth)  + " Tr "+ str(tr) +  " train score: " + str(clf.score(X_train, y_train)) )
                logger.log("max_leaf " + str(max_leaf_node) + " max_depth " + str(max_depth)  + " Tr "+ str(tr) +  " test  score: " + str(test_score) )
                if test_score > test_score_save:
                    save_clf = copy.deepcopy(clf)
    clf = copy.deepcopy(save_clf)
    logger.log(clf)

    y_hat    = clf.predict(X_test)
    y_hat_pr = clf.predict_proba(X_test)[:, 1]

    return clf, y_hat, y_hat_pr