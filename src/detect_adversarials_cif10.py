print('Load modules...')
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
import argparse

import pdb

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='std', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--detector", default='LayerMFS', help="the detector youz want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
parser.add_argument("--net", default='cif10', help="the network used for the attack, either cif10 or cif100")
parser.add_argument("--mode", default='test', help="choose test or validation case")
parser.add_argument("--nr", default='0', help="layer_nr")
parser.add_argument("--wanted_samples", default='1500', type=int, help="wanted_samples")

parser.add_argument("--clf", default='LR', help="LR or RF")



# parser.add_argument("--eps", default='-1',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps", default='0.03137254901960784', type=float,  help="epsilon: 8/255")
# parser.add_argument("--eps", default='0.01568627450980392', type=float, help="epsilon: 4/255")
# parser.add_argument("--eps", default='0.00784313725490196', type=float, help="epsilon: 2/255")
# parser.add_argument("--eps", default='0.00392156862745098', type=float, help="epsilon: 1/255")
parser.add_argument("--eps", default='0.00196078431372549', type=float, help="epsilon: 0.5/255")

args = parser.parse_args()
print(args)

#choose attack
attack_method = args.attack
detector = args.detector
mode = args.mode
net = args.net
scale = True

# import pdb; pdb.set_trace()

#load characteristics
print('Loading characteristics...')

layer_name = [
              'conv2_0WB', 'conv2_1WB', 'conv2_2WB', 'conv2_3WB',
              'conv3_0WB', 'conv3_1WB', 'conv3_2WB', 'conv3_3WB',
              'conv4_0WB', 'conv4_1WB', 'conv4_2WB', 'conv4_3WB',
              'almost_last'
            ]

# save_img = './data/characteristics/layers/'+net+'_'+attack_method+'_'+detector+'_'+layer_name[int(args.nr)]
# save_img = './data/characteristics/layers/'+net+'_'+attack_method+'_'+detector+'_'+layer_name[int(args.nr)]
# save_img = './data/characteristics/0.03137/'+net+'_'+attack_method+'_'+detector
# save_img = './data/characteristics/0.03137/layers/'+net+'_'+attack_method+'_'+detector+'_'+layer_name[int(args.nr)]

# eps = 0.1
# eps = 8/255
# eps = 0.01
# eps = 0.001

# print("eps: ", eps)

# save_img = './data/characteristics/std/' + net + '_' + attack_method + '_' + detector  + '_eps_{:5f}'.format(eps)

# save_img = './data/characteristics/std/' + net+'_'+attack_method+'_'+detector + '_gray' + '_eps_{:5f}'.format(eps)

# if eps == 0.01:
#     save_img = './data/characteristics/std/' + net+'_'+attack_method+'_'+detector + '_5000' + '_eps_{:5f}'.format(eps)

# if eps == 0.001:
#     save_img = './data/characteristics/std/' + net+'_'+attack_method+'_'+detector + '_10000' + '_eps_{:5f}'.format(eps)


# print(save_img)

# characteristics     = pickle.load(open(save_img +     ".p", "rb"))
# characteristics_adv = pickle.load(open(save_img + "_adv.p", "rb"))


if attack_method == 'std':
    print("args eps: ", args.eps)
    if args.detector == 'LayerMFS':
        if args.eps >= 0.03137254901960780:
            images_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.031373.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.031373_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps >= 0.0156862745098039:
            images_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.015686.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.015686_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps >= 0.0078431372549019:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.007843.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.007843_adv.p'
            

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps >= 0.0039215686274509:
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_12000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_12000_Linf_sorted_pert"
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.003922.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.003922_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_6000_Linf_sorted_pert"
            nr_samples = 6000

        elif args.eps >= 0.0019607843137254:
            # images_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.001961.p'
            # images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.001961_adv.p'
            
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_15000_eps_0.001961.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_15000_eps_0.001961_adv.p'

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_pert"
            # nr_samples = 6000
            nr_samples = 15000

    else:
        if args.eps >= 0.03137254901960780:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.031373.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.031373_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_pert"
            nr_samples = 6000 
        elif args.eps >= 0.0156862745098039:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.015686.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.015686_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps >= 0.0078431372549019:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.007843.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.007843_adv.p'
            

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps >= 0.0039215686274509:
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_12000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_12000_Linf_sorted_pert"
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.003922.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_6000_eps_0.003922_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_6000_Linf_sorted_pert"
            nr_samples = 6000

        elif args.eps >= 0.0019607843137254:
            # images_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.001961.p'
            # images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_6000_eps_0.001961_adv.p'
            
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_15000_eps_0.001961.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_InputMFS_15000_eps_0.001961_adv.p'

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_pert"
            # nr_samples = 6000
            nr_samples = 15000


characteristics     = pickle.load(open(images_path,      "rb"))[:args.wanted_samples]
characteristics_adv = pickle.load(open(images_advs_path, "rb"))[:args.wanted_samples]

# characteristics = np.load(save_img + '.npy', allow_pickle=True)
# characteristics_adv = np.load(save_img + '_adv.npy', allow_pickle=True)

# characteristics = np.load('./data/characteristics/'+net+'_'+attack_method+'_'+detector+'.npy', allow_pickle=True)
# characteristics_adv = np.load('./data/characteristics/'+net+'_'+attack_method+'_'+detector+'_adv.npy', allow_pickle=True)

shape = np.shape(characteristics)
k = shape[0]

test_size = 0.2

adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_test = train_test_split(characteristics_adv, np.ones(k), test_size=test_size, random_state=42)
b_X_train_val, b_X_test, b_y_train_val, b_y_test = train_test_split(characteristics, np.zeros(k), test_size=test_size, random_state=42)
adv_X_train, adv_X_val, adv_y_train, adv_y_val = train_test_split(adv_X_train_val, adv_y_train_val, test_size=test_size, random_state=42)
b_X_train, b_X_val, b_y_train, b_y_val = train_test_split(b_X_train_val, b_y_train_val, test_size=test_size, random_state=42)


print("b_X_train",   b_X_train.shape)
print("adv_X_train", adv_X_train.shape)



X_train = np.concatenate(( b_X_train,adv_X_train))
y_train = np.concatenate(( b_y_train,adv_y_train))

if mode == 'test':
    X_test = np.concatenate(( b_X_test, adv_X_test))
    y_test = np.concatenate(( b_y_test,adv_y_test))
elif mode == 'validation':
    X_test = np.concatenate(( b_X_val, adv_X_val))
    y_test = np.concatenate(( b_y_val,adv_y_val))
else:
    print('Not a valid mode')


print("b_X_test",   b_X_test.shape)
print("adv_X_test", adv_X_test.shape)




# train classifier
print('Training classifier...')

# special case
# if (detector == 'LayerMFS'or detector =='LayerPFS') and net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
#     from cuml.svm import SVC
#     scaler  = MinMaxScaler().fit(X_train)
#     X_train = scaler.transform(X_train)
#     X_test  = scaler.transform(X_test)
#     if detector == 'LayerMFS':
#         gamma = 0.1
#         if attack_method == 'cw':
#             C=1
#         else:
#             C=10
#     else:
#         C=10
#         gamma = 0.01
#     clf = SVC(probability=True, C=C, gamma=gamma)
# else:
#     clf = LogisticRegression() # normal case


# C_list = [1, 5, 10, 15, 20]
# gamma_list = [0.001, 0.01, 0.1, 1]


# gamma_list = ['scale', 'rbf', 'poly', 'sigmoid']

# C = 1
# # gamma = 0.01

# C_list = [1, 5, 10, 15]
# clf = svm.SVC(probability=True, C=C_list[0] ) 
# for c in C_list:
#     # for g in gamma_list:
#     clf.set_params(C=c)
#     clf.fit(X_train,y_train)
#     print ("C ", c, "train error: ", clf.score(X_train, y_train) )
#     print ("C ", c, "test error:  ", clf.score(X_test, y_test) )
    
# # print(clf)
# if eps == 0.001:
#     clf = LogisticRegression(max_iter=1500)
# else: 
#     clf = LogisticRegression()

if args.clf == 'LR':
    clf = LogisticRegression()
    print(clf)
    clf.fit(X_train,y_train)
    print ("train error: ", clf.score(X_train, y_train) )
    print ("test error:  ", clf.score(X_test, y_test) )

if args.clf == 'RF':
    # # tress = [100, 200, 300, 400, 500]
    trees = [ 300 ]
    clf = RandomForestClassifier(n_estimators=100)

    for tr in trees:
        # for g in gamma_list:
        clf.set_params(n_estimators=tr)
        print(clf)
        clf.fit(X_train, y_train)
        print ("Tr ", tr, "train error: ", clf.score(X_train, y_train) )
        print ("Tr ", tr, "test error:  ", clf.score(X_test, y_test) )


# save classifier
filename = './data/detectors/LR_'+attack_method+'_'+detector+'_'+mode+'_'+net+'.sav'
pickle.dump(clf, open(filename, 'wb'))

print('Evaluating classifier...')
prediction = clf.predict(X_test)
prediction_pr = clf.predict_proba(X_test)[:, 1]

nr_not_detect_adv = 0

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

    if y_test[i] == 1:
        if prediction[i] == 0:
            nr_not_detect_adv  +=1


acc = (benign_rate+ad_rate)/len(prediction)
TP = 2*ad_rate/len(prediction)
TN = 2*benign_rate/len(prediction)

precision = ad_rate/ad_guesses

TPR = 2*ad_rate/len(prediction)
recall = round(100*TPR,2)

prec = precision 
rec = TPR 

auc = round(100*roc_auc_score(y_test,prediction_pr), 2)
acc = round(100*acc,2)
pre = round(100*precision,1)
tpr = round(100*TP,2)
f1 =  round( 2 * (prec*rec) / (prec+rec), 4)

print('F1-Measure: ', f1)

print('True positive rate/adversarial detetcion rate/recall/sensitivity is ', tpr)
print('True negative rate/normal detetcion rate/selectivity is ', round(100*TN,2))
print('Precision', pre)
print('The accuracy is', acc)
print('The AUC score is', auc)

print('ASRD', 100-acc)

print('len prediction', len(prediction))
print('nr_not_detect_adv', nr_not_detect_adv)

print(auc, acc, pre, tpr, f1)
