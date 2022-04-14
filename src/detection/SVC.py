#!/usr/bin/env python3

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def calc_scaler(X_train, X_test, typ='minmax'):
    
    if typ == 'minmax':
        scaler = MinMaxScaler().fit(X_train)
    elif typ == 'std':
        scaler = StandardScaler().fit(X_train)
        
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    
    return X_train, X_test


def calc_params(args):
    C = 1
    gamma = 'auto'
    if args.detector == 'LayerMFS':
        gamma = 0.1
        if args.attack == 'cw':
            C=1
        else:
            C=10
    else:
        C=10
        gamma = 0.01
        
    return C, gamma


def SVC(args, logger, X_train, y_train, X_test, y_test):
    from sklearn.svm import SVC as SVC_cpu
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
    X_train, X_test = calc_scaler(X_train, X_test, typ='minmax')
    C, gamma = calc_params(args)

    clf = SVC_cpu(probability=True, C=C, gamma=gamma, kernel=args.kernel, max_iter=-1)
    # clf = SVM(kernel=args.kernel, max_iter=args.num_iter)

    logger.log(clf)
    clf.fit(X_train, y_train)
    logger.log("train score: " + str(clf.score(X_train, y_train)) )
    logger.log("test score:  " + str(clf.score(X_test,  y_test)) )

    y_hat =    clf.predict(X_test)
    y_hat_pr = clf.predict_proba(X_test)[:, 1]

    return clf, y_hat, y_hat_pr


def cuSVC(args, logger, X_train, y_train, X_test, y_test):
    from cuml.svm import SVC as SVC_gpu
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
    X_train, X_test = calc_scaler(X_train, X_test, typ='minmax')
    C, gamma = calc_params(args)
    
    clf = SVC_gpu(probability=True, C=C, gamma=gamma, kernel=args.kernel, max_iter=-1)

    # import pdb; pdb.set_trace()

    logger.log(clf)
    clf.fit(X_train, y_train)
    logger.log("train score: " + str(clf.score(X_train, y_train)) )
    logger.log("test score:  " + str(clf.score(X_test,  y_test)) )

    y_hat =    clf.predict(X_test)
    y_hat_pr = clf.predict_proba(X_test)[:, 1]

    return clf, y_hat, y_hat_pr


# clf_svc = SVC(kernel='poly', degree=2, gamma='auto', C=1)
# sklearn_time_svc = %timeit -o train_data(clf_svc)

# clf_svc = SVC_gpu(kernel='poly', degree=2, gamma='auto', C=1)
# cuml_time_svc = %timeit -o train_data(clf_svc)

# print(f"""Average time of sklearn's {clf_svc.__class__.__name__}""", sklearn_time_svc.average, 's')
# print(f"""Average time of cuml's {clf_svc.__class__.__name__}""", cuml_time_svc.average, 's')

# print('Ratio between sklearn and cuml is', sklearn_time_svc.average/cuml_time_svc.average)