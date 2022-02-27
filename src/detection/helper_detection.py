#!/usr/bin/env python3

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def show_results(args, logger, y_test, y_hat, y_hat_pr):

    nr_not_detect_adv = 0

    benign_rate = 0
    benign_guesses = 0
    ad_guesses = 0
    ad_rate = 0
    for i in range(len(y_hat)):
        if y_hat[i] == 0:
            benign_guesses +=1
            if y_test[i]==0:
                benign_rate +=1
        else:
            ad_guesses +=1
            if y_test[i]==1:
                ad_rate +=1

        if y_test[i] == 1:
            if y_hat[i] == 0:
                nr_not_detect_adv  +=1

    acc = (benign_rate+ad_rate)/len(y_hat)        
    TP = 2*ad_rate/len(y_hat)
    TN = 2*benign_rate/len(y_hat)

    precision = ad_rate/ad_guesses

    TPR = 2 * ad_rate / len(y_hat)
    recall = round(100*TPR, 2)

    prec = precision 
    rec = TPR 


    auc = round(100*roc_auc_score(y_test, y_hat_pr), 2)
    acc = round(100*acc, 2)
    pre = round(100*precision, 1)
    tpr = round(100*TP, 2)
    f1  = round((2 * (prec*rec) / (prec+rec))*100, 2)
    fnr = round(100 - tpr, 2)

    logger.log('F1-Measure: ' + str(f1) )
    logger.log('PREC: ' + str(pre) )
    logger.log('ACC: ' + str(acc) )
    logger.log('AUC: ' + str(auc) )
    logger.log('TNR: ' + str(round(100*TN, 2)) ) # True negative rate/normal detetcion rate/selectivity is 
    logger.log('TPR: ' + str(tpr) )# True positive rate/adversarial detetcion rate/recall/sensitivity is 
    logger.log('FNR: ' + str(fnr) )
    logger.log('RES:, AUC, ACC, PRE, TPR, F1, FNR' )
    logger.log('RES:,' + str(auc) + ',' + str(acc) + ',' + str(pre) + ',' + str(tpr) + ',' + str(f1) + ',' + str(fnr) )
    logger.log('<==========================================================================')

    # tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=list(range(args.num_classes))).ravel()
    # tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=list(range(args.num_classes))).ravel()

    # aa, bb, cc, dd = perf_measure(y_test, y_hat)

    # fpr, tpr, _ = roc_curve(y_test, y_hat_pr)
    # print("fpr", fpr)
    # print("tpr", tpr)
