#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def LR(args, logger, X_train, y_train, X_test, y_test):
    
    # penalty {‘l1", ‘l2", ‘elasticnet", ‘none"}, default="l2"
    # dual bool, default=False --> primal
    # tolf loat, default=1e-4
    # C float, default=1.0
    # intercept_scaling float, default=1
    # class_weightdict or ‘balanced", default=None
    # random_stateint, RandomState instance, default=None
    # max_iter int, default=100
    # multi_class{‘auto", ‘ovr", ‘multinomial"}, default="auto"
    
    # scaler  = MinMaxScaler().fit(X_train)
    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)


    if args.tuning == None:

        clf = LogisticRegression(max_iter=args.num_iter, random_state=21)

        logger.log(clf)
        clf.fit(X_train, y_train)
        logger.log("train score: " + str(clf.score(X_train, y_train)) )
        logger.log("test score:  " + str(clf.score(X_test,  y_test)) )

    elif args.tuning in ["randomsearch"]:
        
        random_grid = {
            "max_iter": [100, 200, 400],
            "C":np.logspace(-3,3,7),
            "penalty":[None, "l1","l2", "elasticnet"], # l1 lasso l2 ridge
            "dual": [False, True],
            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        }

        lr = LogisticRegression(random_state=21, n_jobs=-1)
        lr_random = RandomizedSearchCV(estimator=lr, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=21, n_jobs=-1)
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
        
        lr_random.fit(X_train, y_train)
        print('Random grid: ', random_grid, '\n')
        
        # print the best parameters
        print('Best Parameters: ', lr_random.best_params_ , ' \n')
        clf = LogisticRegression(**lr_random.best_params_, random_state=21, n_jobs=-1)
        clf.fit(X_train, y_train)


    elif args.tuning in ["gridsearch"]:
        random_grid = {
            "max_iter": [100, 400, 800],
            "C":np.logspace(-3,3,7),
            "penalty":[None, "l1","l2", "elasticnet"], # l1 lasso l2 ridge
            "dual": [False, True],
            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        }
 

        lr = LogisticRegression(random_state=21, n_jobs=-1)
        lr_random = GridSearchCV(estimator=lr, param_grid=random_grid, 
                                #scoring=scoring, 
                                refit="AUC", cv=3, verbose=2, n_jobs=-1)

        lr_random.fit(X_train, y_train)

        print('Random grid: ', random_grid, '\n')
        
        # print the best parameters
        print('Best Parameters: ', lr_random.best_params_ , ' \n')
        clf = LogisticRegression(**lr_random.best_params_, random_state=21, n_jobs=-1)
        clf.fit(X_train, y_train)

    y_hat    = clf.predict(X_test)
    y_hat_pr = clf.predict_proba(X_test)[:, 1]

    return clf, y_hat, y_hat_pr