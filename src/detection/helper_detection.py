#!/usr/bin/env python3

from conf import settings

import torch
import os, pdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

import  numpy as np
import time



def perf_measure(y_actual, y_hat):
    """
    https://shouland.com/false-positive-rate-test-sklearn-code-example
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return (TP, FP, TN, FN)


def other(prediction, prediction_pr, y_test):
    benign_rate = 0
    benign_guesses = 0
    ad_guesses = 0
    ad_rate = 0
    for i in range(len(prediction)):
        if prediction[i] == 0:
            benign_guesses +=1
            if y_test[i]==0:
                benign_rate +=1
        else:
            ad_guesses +=1
            if y_test[i]==1:
                ad_rate +=1

    acc = (benign_rate+ad_rate)/len(prediction)        
    TP = 2*ad_rate/len(prediction)
    TN = 2*benign_rate/len(prediction)

    print('True positive rate/adversarial detetcion rate/recall/sensitivity is ', round(100*TP,2))
    print('True negative rate/normal detetcion rate/selectivity is ', round(100*TN,2))
    print('The accuracy is',round(100*acc,2))
    print('The AUC score is', round(100*roc_auc_score(y_test,prediction_pr),2))


def show_results(args, logger, y_actual, y_hat, y_hat_pr):
    
    print("len(y_actual)",   len(y_actual) )
    print("len(y_hat)",    len(y_hat) )
    print("len(y_hat_pr)", len(y_hat_pr) )
    
    TP, FP, TN, FN = perf_measure(y_actual, y_hat)
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP) 
    FNR = FN / (FN + TP)
    
    # ACC = (TP + TN) / (TP + TN + FP + FN)
    # PRECISION = TP / (TP + FP)
    # F1 =  2 * (PRECISION*TPR) / (PRECISION+TPR)
    ACC = accuracy_score(y_actual, y_hat)
    PRECISION = precision_score(y_actual, y_hat)
    F1 = f1_score(y_actual, y_hat)
    
    if args.clf == 'IF': 
        auc = -1
    else: 
        auc = round(100*roc_auc_score(y_actual, y_hat_pr), 2)
        
    f1  = round(100*F1, 2)
    pre = round(100*PRECISION, 2)
    acc = round(100*ACC, 2)
    tpr = round(100*TPR, 2)
    tnr = round(100*TNR, 2)
    fnr = round(100*FNR, 2)

    logger.log('F1:   ' +  str(f1) )
    logger.log('PREC: ' +  str(pre) )
    logger.log('ACC:  ' +  str(acc) )
    logger.log('AUC:  ' +  str(auc) )
    logger.log('TPR:  ' +  str(tpr) ) # True positive rate/adversarial detetcion rate/recall/sensitivity is 
    logger.log('TNR:  ' +  str(tnr) ) # True negative rate/normal detetcion rate/selectivity is 
    logger.log('FNR:  ' +  str(fnr) )
    
    # logger.log('RES:, AUC, ACC, PRE, TPR, F1, FNR' )
    # logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(fnr) )

    logger.log('RES:, AUC, ACC, PRE, TPR, F1, TNR, FNR' )
    logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(tnr) + ',' + str(fnr) )
    # logger.log('RES:, AUC, ACC, PRE, TPR, F1, FNR' )
    # logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(fnr) )
    logger.log('<==========================================================================')


def show_results_attack_transfer(args, logger, y_actual, y_hat, y_hat_pr, X_train, y_train, y_train_pr):
    
    print("len(y_actual)",   len(y_actual) )
    print("len(y_hat)",    len(y_hat) )
    print("len(y_hat_pr)", len(y_hat_pr) )
    
    # other(y_hat, y_hat_pr, y_actual)
    
    fpr, tpr, thresholds = roc_curve(y_train, y_train_pr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.argmax( np.sqrt((tpr)**2 + (1-fpr)**2 ))
    optimal_threshold = thresholds[optimal_idx]
    y_hat = np.ones_like(y_hat)
    y_hat[np.where( y_hat_pr[y_hat_pr <= optimal_threshold] )]  = 0
    # y_hat[np.where(y_hat_pr[y_hat_pr < optimal_threshold] )] = 0
    
    # import pdb; pdb.set_trace()
    
    TP, FP, TN, FN = perf_measure(y_actual, y_hat)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP) 
    FNR = FN / (FN + TP)
    
    # ACC = (TP + TN) / (TP + TN + FP + FN)
    ACC = accuracy_score(y_actual, y_hat)    
    # PRECISION = TP / (TP + FP)
    # PRECISION = precision_score(y_actual, y_hat, average='weighted', labels=np.unique(y_hat_pr))

    PRECISION = precision_score(y_actual, y_hat)
    RECALL    = recall_score(y_actual, y_hat)
    
    # F1 =  2 * (PRECISION*TPR) / (PRECISION+TPR)
    F1 = f1_score(y_actual, y_hat)
    
    # import pdb; pdb.set_trace()
    # PRECISION = 0.1

    if args.clf == 'IF': 
        auc = -1
    else: 
        auc = round(100*roc_auc_score(y_actual, y_hat_pr), 2)
        
    f1  = round(100*F1, 2)
    pre = round(100*PRECISION, 2)
    rec = round(100*RECALL, 2)
    acc = round(100*ACC, 2)
    tpr = round(100*TPR, 2)
    tnr = round(100*TNR, 2)
    fnr = round(100*FNR, 2)

    logger.log('F1:   ' +  str(f1) )
    logger.log('PREC: ' +  str(pre) )
    logger.log('RECA: ' +  str(rec) )
    logger.log('ACC:  ' +  str(acc) )
    logger.log('AUC:  ' +  str(auc) )
    logger.log('TPR:  ' +  str(tpr) ) # True positive rate/adversarial detetcion rate/recall/sensitivity is 
    logger.log('TNR:  ' +  str(tnr) ) # True negative rate/normal detetcion rate/selectivity is 
    logger.log('FNR:  ' +  str(fnr) )
    
    # logger.log('RES:, AUC, ACC, PRE, TPR, F1, FNR' )
    # logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(fnr) )

    logger.log('RES:, AUC, ACC, PRE, TPR, F1, TNR, FNR' )
    logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(tnr) + ',' + str(fnr) )
    # logger.log('RES:, AUC, ACC, PRE, TPR, F1, FNR' )
    # logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(fnr) )
    logger.log('<==========================================================================')
    


def split_data(args, logger, characteristics, characteristics_adv, noise=False, test_size=0.2, random_state=42):
    
    shape_adv = np.shape(characteristics_adv)[0]
    shape_char = np.shape(characteristics)[0]
    
    # import pdb; pdb.set_trace()
    # adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_actual = train_test_split(characteristics_adv, np.ones(shape_adv),   test_size=test_size, random_state=random_state)
    # b_X_train_val, b_X_test, b_y_train_val, b_y_actual         = train_test_split(characteristics,     np.zeros(shape_char), test_size=test_size, random_state=random_state)
    # adv_X_train, adv_X_val, adv_y_train, adv_y_val             = train_test_split(adv_X_train_val,     adv_y_train_val,      test_size=test_size, random_state=random_state)
    # b_X_train, b_X_val, b_y_train, b_y_val                     = train_test_split(b_X_train_val,       b_y_train_val,        test_size=test_size, random_state=random_state)

    adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_test = train_test_split(characteristics_adv, np.ones(shape_adv), test_size=0.2, random_state=42)
    b_X_train_val, b_X_test, b_y_train_val, b_y_test = train_test_split(characteristics, np.zeros(shape_char), test_size=0.2, random_state=42)
    adv_X_train, adv_X_val, adv_y_train, adv_y_val = train_test_split(adv_X_train_val, adv_y_train_val, test_size=0.2, random_state=42)
    b_X_train, b_X_val, b_y_train, b_y_val = train_test_split(b_X_train_val, b_y_train_val, test_size=0.2, random_state=42)

    # if not noise:
    #     adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_actual = train_test_split(characteristics_adv, np.ones(shape_adv),   test_size=test_size, random_state=random_state)
    #     b_X_train_val, b_X_test, b_y_train_val, b_y_actual         = train_test_split(characteristics,     np.zeros(shape_char), test_size=test_size, random_state=random_state)
    #     adv_X_train, adv_X_val, adv_y_train, adv_y_val             = train_test_split(adv_X_train_val,     adv_y_train_val,      test_size=test_size, random_state=random_state)
    #     b_X_train, b_X_val, b_y_train, b_y_val                     = train_test_split(b_X_train_val,       b_y_train_val,        test_size=test_size, random_state=random_state)
    # else:
    #     adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_actual = train_test_split(characteristics_adv,    np.ones(shape_adv),  test_size=test_size, random_state=random_state)
    #     b_X_train_val, b_X_test, b_y_train_val, b_y_actual         = train_test_split(characteristics[:shape_adv], np.zeros(shape_adv), test_size=test_size, random_state=random_state)
    #     # b_X_train_val3, b_X_test3, b_y_train_val3, b_y_actual3     = train_test_split(characteristics[:2000], np.zeros(shape_adv), test_size=test_size, random_state=random_state)
    #     b_X_train_val2, b_X_test2, b_y_train_val2, b_y_actual2     = train_test_split(characteristics[shape_adv:], np.zeros(shape_adv), test_size=test_size, random_state=random_state)
        
    #     b_X_train_val = np.concatenate((b_X_train_val, b_X_train_val2))
    #     b_X_test = np.concatenate((b_X_test, b_X_test2))
    #     b_y_train_val = np.concatenate((b_y_train_val, b_y_train_val2))
    #     b_y_actual = np.concatenate((b_y_actual, b_y_actual2)) 
        
    #     adv_X_train, adv_X_val, adv_y_train, adv_y_val           = train_test_split(adv_X_train_val,     adv_y_train_val,      test_size=test_size, random_state=random_state)
    #     b_X_train, b_X_val, b_y_train, b_y_val                   = train_test_split(b_X_train_val,       b_y_train_val,        test_size=test_size, random_state=random_state)


    # X_train = np.concatenate(( b_X_train, adv_X_train) )
    # y_train = np.concatenate(( b_y_train, adv_y_train) )

    # X_test = np.concatenate( (b_X_test, adv_X_test, b_X_val, adv_X_val) )
    # y_actual = np.concatenate( (b_y_actual, adv_y_actual, b_y_val, adv_y_val) )
    
    X_train = np.concatenate(( b_X_train,adv_X_train))
    y_train = np.concatenate(( b_y_train,adv_y_train))

    X_test = np.concatenate(( b_X_test, adv_X_test))
    y_actual = np.concatenate(( b_y_test,adv_y_test))

    # if args.mode == 'test':
    #     X_test = np.concatenate( (b_X_test, adv_X_test) )
    #     y_actual = np.concatenate( (b_y_actual, adv_y_actual) )
    # elif args.mode == 'validation':
    #     X_test = np.concatenate( (b_X_val, adv_X_val) )
    #     y_actual = np.concatenate( (b_y_val, adv_y_val) )
    # else:
    #     logger.log('Not a valid mode')

    logger.log("b_X_train" + str(b_X_train.shape) )
    logger.log("adv_X_train" + str(adv_X_train.shape) )

    logger.log("b_X_test" + str(b_X_test.shape) )
    logger.log("adv_X_test" + str(adv_X_test.shape) )

    return X_train, y_train, X_test, y_actual


def save_load_clf(args, clf, output_path_dir):
    # save classifier
    classifier_pth = output_path_dir + os.sep + str(args.clf) + '.clf'
    if settings.SAVE_CLASSIFIER:
        torch.save(clf, classifier_pth)
    else:
        clf = torch.load(classifier_pth)
    
    return clf


def compute_time_sample(args, clf, X_train, y_train, X_test, y_actual):
    
    print("compute time sample")
    print("clf: ", clf)
    
    NR_LOOPS = 100
    WARM_UP = 10
    nr_samples = 100
    compute_time_sample = []
    
    for loop in range(NR_LOOPS + WARM_UP):
        tstart = time.time() * 1000.0
        # import pdb; pdb.set_trace()
        y_hat = clf.predict(X_test[:nr_samples])
        tend = time.time() * 1000.0
        difference = tend - tstart
        if loop >= WARM_UP:
            compute_time_sample.append(difference / nr_samples)
    
    np_compute_time_sample     = np.asarray(compute_time_sample)
    StdDev = np.std(np_compute_time_sample)
    Mean = np.mean(np_compute_time_sample)
    print("NR_LOOPS:   ", NR_LOOPS)
    print("WarmUp:     ", WARM_UP)
    print("OneSample:    {}".format(np_compute_time_sample[50]))
    print("Mean:       ", Mean )
    print("StdDev:     ", StdDev )
    
    print("NR_LOOPS, WARM_UP, OneSample, Mean, StdDev")
    print(NR_LOOPS, WARM_UP, np_compute_time_sample[50], Mean, StdDev)
    print("==========================================================================")
