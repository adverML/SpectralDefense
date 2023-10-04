#!/usr/bin/env python3


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
import copy
import pdb
import numpy as np


def RF(args, logger, X_train, y_train, X_test, y_test):
    

    if args.tuning == None:
    
        max_depths     = [None]
        max_leaf_nodes = [None]
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
        
        # if settings.SAVE_CLASSIFIER:
        #     save_load_clf(args, clf, output_path_dir)
        # else:
        #     clf = save_load_clf(args, clf, output_path_dir)

    
    elif args.tuning in ["randomsearch"]:
        
        random_grid = {
            'n_estimators': [50,100, 300, 500, 800], # number of trees in the random forest
            'max_features': ['auto', 'sqrt', 'log2'], # number of features in consideration at every split
            'max_depth': [int(x) for x in np.linspace(10, 120, num = 12)], # maximum number of levels allowed in each decision tree,
            'min_samples_split': [2, 6, 10], # minimum sample number to split a node
            'min_samples_leaf':  [1, 3, 4],  # minimum sample number that can be stored in a leaf node
            'bootstrap': [True, False],
            #'criterion' : ['gini', 'entropy'],
        }
 
        scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)} # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html

        rf = RandomForestRegressor(random_state=21,  n_jobs=-1)
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=21, n_jobs=-1)
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
        
        rf_random.fit(X_train, y_train)
        print('Random grid: ', random_grid, '\n')
        
        # print the best parameters
        print('Best Parameters: ', rf_random.best_params_ , ' \n')
        clf = RandomForestClassifier(**rf_random.best_params_, random_state=21, n_jobs=-1)
        clf.fit(X_train, y_train)


    elif args.tuning in ["gridsearch"]:
        random_grid = {
            'n_estimators': [50,100, 300, 500, 800], # number of trees in the random forest
            'max_features': ['auto', 'sqrt', 'log2'], # number of features in consideration at every split
            'max_depth': [int(x) for x in np.linspace(10, 120, num=12)], # maximum number of levels allowed in each decision tree,
            'min_samples_split': [2, 6, 10], # minimum sample number to split a node
            'min_samples_leaf':  [1, 3, 4],  # minimum sample number that can be stored in a leaf node
            'bootstrap': [True, False],
            #'criterion' : ['gini', 'entropy'],
        }

        # scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}
        rf = RandomForestRegressor(random_state=21, n_jobs=-1)
        rf_random = GridSearchCV(estimator=rf, param_grid=random_grid, 
                                #scoring=scoring, 
                                refit="AUC", cv=3, verbose=2, n_jobs=-1)

        rf_random.fit(X_train, y_train)

        print ('Random grid: ', random_grid, '\n')
        
        # print the best parameters
        print ('Best Parameters: ', rf_random.best_params_ , '\n')

        clf = RandomForestClassifier(**rf_random.best_params_, random_state=21, n_jobs=-1) 
        clf.fit(X_train, y_train) 

    y_hat    = clf.predict(X_test)
    y_hat_pr = clf.predict_proba(X_test)[:, 1]

    return clf, y_hat, y_hat_pr