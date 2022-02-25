print('Load modules...')
import numpy as np
import pickle
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn import svm
import argparse

import copy

import pdb

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack",   default='std', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--detector", default='LayerMFS', help="the detector youz want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
parser.add_argument("--net",  default='imagenet32', help="the network used for the attack, either cif10 or cif100")
parser.add_argument("--mode", default='test', help="choose test or validation case")
parser.add_argument("--nr",   default='0', help="layer_nr")
parser.add_argument("--wanted_samples", default='1000', type=int, help="wanted_samples")
parser.add_argument("--clf",  default='LR', help="LR or RF")
parser.add_argument('--img_size', type=int, default=32)

# parser.add_argument("--eps", default='-1',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
parser.add_argument("--eps",  default='0.03137254901960784', type=float,  help="epsilon: 8/255")
# parser.add_argument("--eps", default='0.01568627450980392', type=float, help="epsilon: 4/255")
# parser.add_argument("--eps", default='0.00784313725490196', type=float, help="epsilon: 2/255")
# parser.add_argument("--eps", default='0.00392156862745098', type=float, help="epsilon: 1/255")
# parser.add_argument("--eps", default='0.00196078431372549', type=float, help="epsilon: 0.5/255")

# parser.add_argument("--eps_test", default='-1',       help="epsilon: 4/255, 3/255, 2/255, 1/255, 0.5/255")
# parser.add_argument("--eps_test", default='0.03137254901960784', type=float,  help="epsilon: 8/255")
# parser.add_argument("--eps_test", default='0.01568627450980392', type=float, help="epsilon: 4/255")
# parser.add_argument("--eps_test", default='0.00784313725490196', type=float, help="epsilon: 2/255")
parser.add_argument("--eps_test", default='0.00392156862745098', type=float, help="epsilon: 1/255")
# parser.add_argument("--eps_test", default='0.00196078431372549', type=float, help="epsilon: 0.5/255")


args = parser.parse_args()
print(args)

#choose attack
attack_method = args.attack
detector = args.detector
mode = args.mode
net = args.net
scale = True

assert not args.eps == args.eps_test, ("Epsilon must be different!")

print


#load characteristics
print('Loading characteristics...')

layer_name = [
              'conv2_0WB', 'conv2_1WB', 'conv2_2WB', 'conv2_3WB',
              'conv3_0WB', 'conv3_1WB', 'conv3_2WB', 'conv3_3WB',
              'conv4_0WB', 'conv4_1WB', 'conv4_2WB', 'conv4_3WB',
              'almost_last'
            ]

nr_img = 0


if attack_method == 'apgd-ce':
    images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_apgd-ce_{}.p'.format(detector)
    images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_apgd-ce_{}_adv.p'.format(detector)

    # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_apgd-ce_LayerMFS.npy"
    # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_apgd-ce_LayerMFS_adv.npy"
    nr_samples = 6000 
elif attack_method == 'apgd-t':
    images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_apgd-t_{}.p'.format(detector)
    images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_apgd-t_{}_adv.p'.format(detector)

    # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_apgd-t_LayerMFS.npy"
    # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_apgd-t_LayerMFS_adv.npy"
    nr_samples = 6000
elif attack_method == 'fab-t':
    images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_fab-t_{}.p'.format(detector)
    images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_fab-t_{}_adv.p'.format(detector)
    # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_fab-t_LayerMFS.npy"
    # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_fab-t_LayerMFS_adv.npy"
    nr_samples = 6000
elif attack_method == 'square':
    images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_square_{}.p'.format(detector)
    images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/ind/imagenet_square_{}_adv.p'.format(detector)
    # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_square_LayerMFS.npy"
    # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/0.03137/imagenet/imagenet_square_LayerMFS_adv.npy"
    nr_samples = 6000


elif attack_method == 'std':
    print("args eps: ", args.eps_test)
    if args.detector == 'LayerMFS':
        if args.eps_test >= 0.03137254901960780:
            if args.img_size == 32:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_LayerMFS_6000_eps_0.031373.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_LayerMFS_6000_eps_0.031373_adv.p'
            elif args.img_size == 64:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_LayerMFS_6000_eps_0.031373.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_LayerMFS_6000_eps_0.031373_adv.p'
            elif args.img_size == 128:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_LayerMFS_6000_eps_0.031373.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_LayerMFS_6000_eps_0.031373_adv.p'

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_pert"
            nr_samples = 6000 
        elif args.eps_test >= 0.0156862745098039:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_LayerMFS_6000_eps_0.015686.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_LayerMFS_6000_eps_0.015686_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps_test >= 0.0078431372549019:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_LayerMFS_6000_eps_0.007843.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_LayerMFS_6000_eps_0.007843_adv.p'
            

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps >= 0.0039215686274509:
            # images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_LayerMFS_6000_eps_0.003922.p'
            # images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_LayerMFS_6000_eps_0.003922_adv.p'
            nr_samples = 6000
            if args.img_size == 32:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_LayerMFS_6000_eps_0.003922.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_LayerMFS_6000_eps_0.003922_adv.p'
            elif args.img_size == 64:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_LayerMFS_6000_eps_0.003922.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_LayerMFS_6000_eps_0.003922_adv.p'
            elif args.img_size == 128:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_LayerMFS_6000_eps_0.003922.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_LayerMFS_6000_eps_0.003922_adv.p'

        elif args.eps >= 0.0019607843137254:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_LayerMFS_6000_eps_0.003922.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_LayerMFS_6000_eps_0.001961_adv.p'
            
            # images_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_15000_eps_0.001961.p'
            # images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_15000_eps_0.001961_adv.p'

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_pert"
            nr_samples = 6000
            # nr_samples = 15000
        
    else:
        if args.eps_test >= 0.03137254901960780:
            # images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.031373.p'
            # images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.031373_adv.p'

            if args.img_size == 32:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.031373.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.031373_adv.p'
            elif args.img_size == 64:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_InputMFS_6000_eps_0.031373.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_InputMFS_6000_eps_0.031373_adv.p'
            elif args.img_size == 128:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_InputMFS_6000_eps_0.031373.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_InputMFS_6000_eps_0.031373_adv.p'
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.031373_std_6000_Linf_sorted_pert"
            nr_samples = 6000 
        elif args.eps_test >= 0.0156862745098039:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_InputMFS_6000_eps_0.015686.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_InputMFS_6000_eps_0.015686_adv.p'
            
            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.015686_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps_test >= 0.0078431372549019:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_InputMFS_6000_eps_0.007843.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_InputMFS_6000_eps_0.007843_adv.p'
            

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.007843_std_6000_Linf_sorted_pert"
            nr_samples = 6000
        elif args.eps_test >= 0.0039215686274509:
            
            # images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.003922.p'
            # images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.003922_adv.p'
            if args.img_size == 32:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.003922.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet32/std/imagenet32_std_InputMFS_6000_eps_0.003922.p'
            elif args.img_size == 64:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_InputMFS_6000_eps_0.003922.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet64/std/imagenet64_std_InputMFS_6000_eps_0.003922_adv.p'
            elif args.img_size == 128:
                images_path =      '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_InputMFS_6000_eps_0.003922.p'
                images_advs_path = '/home/lorenzp/adversialml/src/src/data/characteristics/imagenet128/std/imagenet128_std_InputMFS_6000_eps_0.003922_adv.p'

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.003922_std_6000_Linf_sorted_pert"
            nr_samples = 6000

        elif args.eps_test >= 0.0019607843137254:
            images_path =      '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_InputMFS_6000_eps_0.001961.p'
            images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/imagenet/std/imagenet_std_InputMFS_6000_eps_0.001961_adv.p'
            
            # images_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_15000_eps_0.001961.p'
            # images_advs_path = '/home/lorenzp/adversialml/src/pytorch-CelebAHQ/data/characteristics/cifar/std/cif10_std_LayerMFS_15000_eps_0.001961_adv.p'

            # images_path =      "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_orig"
            # images_advs_path = "/home/lorenzp/adversialml/src/pytorch-CelebAHQ/auto-attack/adv_complete/std/0.001961_std_6000_Linf_sorted_pert"
            nr_samples = 6000
            # nr_samples = 15000


else:
    path = './data/characteristics/' + net + '/' + net + '_' + attack_method + '_' + detector
    print("open file: ", path)

    images_path =       path +     '.p'
    images_advs_path =  path + '_adv.p'


# characteristics     = torch.load(images_path     )  #[:args.wanted_samples]
# characteristics_adv = torch.load(images_advs_path) #[:args.wanted_samples]

# characteristics     = np.load(images_path     , allow_pickle=True)  #[:args.wanted_samples]
# characteristics_adv = np.load(images_advs_path, allow_pickle=True) #[:args.wanted_samples]

print("images_path:      ", images_path)
print("images_advs_path: ", images_advs_path)

characteristics     = pickle.load(open(images_path, "rb"))[:args.wanted_samples]
characteristics_adv = pickle.load(open(images_advs_path, "rb"))[:args.wanted_samples]

shape = np.shape(characteristics)
k = shape[0]

test_size = 0.2

adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_test = train_test_split(characteristics_adv, np.ones(k), test_size=test_size, random_state=42)
b_X_train_val, b_X_test, b_y_train_val, b_y_test         = train_test_split(characteristics, np.zeros(k), test_size=test_size, random_state=42)
adv_X_train, adv_X_val, adv_y_train, adv_y_val           = train_test_split(adv_X_train_val, adv_y_train_val, test_size=test_size, random_state=42)
b_X_train, b_X_val, b_y_train, b_y_val                   = train_test_split(b_X_train_val, b_y_train_val, test_size=test_size, random_state=42)

X_train = np.concatenate(( b_X_train, adv_X_train))
y_train = np.concatenate(( b_y_train, adv_y_train))

if mode == 'test':
    X_test = np.concatenate(( b_X_test, adv_X_test))
    y_test = np.concatenate(( b_y_test, adv_y_test))
elif mode == 'validation':
    X_test = np.concatenate(( b_X_val, adv_X_val))
    y_test = np.concatenate(( b_y_val, adv_y_val))
else:
    print('Not a valid mode')

print("b_X_train",   b_X_train.shape)
print("adv_X_train", adv_X_train.shape)

print("b_X_test",   b_X_test.shape)
print("adv_X_test", adv_X_test.shape)


#train classifier
print('Training classifier...')


#special case
# if (detector == 'LayerMFS'or detector =='LayerPFS') and net == 'imagenet33' and (attack_method=='std' or attack_method=='cw' or attack_method=='df'):
#     print("SVM")
#     # from cuml.svm import SVC
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
#     # clf = SVC(probability=True, C=C, gamma=gamma)
#     clf = svm.SVC(probability=True, C=C, gamma=gamma ) # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# else:
#     # clf = RandomForestClassifier(max_depth=3, n_estimators=300)


# C_list = [1, 5, 10, 15, 20]
# gamma_list = [0.001, 0.01, 0.1, 1]

# clf = LogisticRegression()

# C_list = [1, 5, 10]
# gamma_list = [0.01, 0.1]

# C = 1
# gamma = 0.01
# clf = svm.SVC(probability=True, C=C, gamma=gamma ) 


# for c in C_list:
#     for g in gamma_list:
#         clf.set_params(C=c, gamma=g )
#         clf.fit(X_train,y_train)
#         print ("C ", c, " gamma", g, "train error: ", clf.score(X_train, y_train) )
#         print ("C ", c, " gamma", g, "test error:  ", clf.score(X_test, y_test) )


# if args.clf == 'LR':
#     clf = LogisticRegression()
#     print(clf)
#     clf.fit(X_train,y_train)
#     print ("train error: ", clf.score(X_train, y_train) )
#     print ("test error:  ", clf.score(X_test, y_test)   )

# if args.clf == 'RF':
#     # trees = [100, 200, 300, 400, 500]
#     # trees = [600, 700, 800, 900]
#     # trees = [ 500 ]
#     trees = [ 300 ]

#     clf = RandomForestClassifier(n_estimators=100)

#     save_clf = copy.deepcopy(clf)
#     test_score_save = 0

#     for tr in trees:
#         clf.set_params(n_estimators=tr)
#         clf.fit(X_train, y_train)

#         test_score = clf.score(X_test, y_test)
#         print ("Tr ", tr, "train error: ", clf.score(X_train, y_train) )
#         print ("Tr ", tr, "test error:  ", test_score)
#         if test_score > test_score_save:
#             save_clf = copy.deepcopy(clf)
#     clf = copy.deepcopy(save_clf)


# save classifier
filename = './data/detectors/'+args.clf+'_'+attack_method+'_'+detector+'_'+mode+'_'+net+'_{:5f}.sav'.format(args.eps)
# if attack_method == 'std':
#     filename = './data/detectors/'+args.clf+'_'+attack_method+'_'+detector+'_'+mode+'_'+net+'_{:5f}.sav'.format(args.eps)
# pickle.dump(clf, open(filename, 'wb'))

if Path(filename).exists():
    with open(filename, "rb") as f:
        # unpickler = pickle.Unpickler(f)
        # pdb.set_trace()
        # clf = pickle.load(f)
        # clf = unpickler.load()
        clf = pickle.load(f)
        print("clf", clf)
else:
    assert False, ("No path exist!", filename)


# pdb.set_trace()

print('Evaluating classifier...')
prediction =    clf.predict(X_test)
prediction_pr = clf.predict_proba(X_test)[:, 1]


print ("train error: ", clf.score(X_train, y_train) )
print ("test error:  ", clf.score(X_test, y_test) )

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

TPR = 2 * ad_rate / len(prediction)
recall = round(100*TPR, 2)

prec = precision 
rec = TPR 

auc = round(100*roc_auc_score(y_test,prediction_pr), 2)
acc = round(100*acc, 2)
pre = round(100*precision, 1)
tpr = round(100*TP, 2)
f1 =  round( 2 * (prec*rec) / (prec+rec), 4)

fnr = 100 - tpr

print('F1-Measure: ', f1)

print('True positive rate/adversarial detetcion rate/recall/sensitivity is ', tpr)
print('True negative rate/normal detetcion rate/selectivity is ', round(100*TN, 2))
print('Precision', pre)
print('The accuracy is', acc)
print('The AUC score is', auc)

print('FNR',  fnr)


print(auc, acc, pre, tpr, f1)
