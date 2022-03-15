#!/usr/bin/env python3
"""
Source: https://github.com/bam098/deep_knn

"""
import pdb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.utils.data as data_utils
import torchvision
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import falconn
import faiss
import platform
import enum
import copy
from bisect import bisect_left
import warnings


from utils import (
    get_num_classes,
    get_debug_info
)

num_epochs = 6            # number of training epochs
batch_size_train = 256    # batch size for training
batch_size_test = 512    # batch size for testing
learning_rate = 0.001     # learning rate for training
calibset_size = 750       # size of the calibration set for DkNN
neighbors = 75            # number of nearest neighbors for DkNN
number_bits = 17          # number of bits for LSH for DkNN

log_interval = 10         # printing training statistics after 10 iterations

criterion = nn.CrossEntropyLoss()

use_cuda = True

class NearestNeighbor:

    class BACKEND(enum.Enum):
        FALCONN = 1
        FAISS = 2

    def __init__(self, backend, dimension, neighbors, number_bits, nb_tables=None):
        assert backend in NearestNeighbor.BACKEND

        self._NEIGHBORS = neighbors
        self._BACKEND = backend

        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._init_falconn(dimension, number_bits, nb_tables)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._init_faiss(dimension)
        else:
            raise NotImplementedError

    def _init_falconn(self, dimension, number_bits, nb_tables):
        assert nb_tables >= self._NEIGHBORS

        # LSH parameters
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = dimension
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = nb_tables
        params_cp.num_rotations = 2  # for dense set it to 1; for sparse data set it to 2
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 0  # we want to use all the available threads to set up
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable

        # we build number_bits-bit hashes so that each table has
        # 2^number_bits bins; a rule of thumb is to have the number
        # of bins be the same order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(number_bits, params_cp)
        self._falconn_table = falconn.LSHIndex(params_cp)
        self._falconn_query_object = None
        self._FALCONN_NB_TABLES = nb_tables

    def _init_faiss(self, dimension):
        res = faiss.StandardGpuResources()
        self._faiss_index = faiss.GpuIndexFlatL2(res, dimension)

    def add(self, x):
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            self._falconn_table.setup(x)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            self._faiss_index.add(x)
        else:
            raise NotImplementedError

    def find_knns(self, x, output):
        if self._BACKEND is NearestNeighbor.BACKEND.FALCONN:
            return self._find_knns_falconn(x, output)
        elif self._BACKEND is NearestNeighbor.BACKEND.FAISS:
            return self._find_knns_faiss(x, output)
        else:
            raise NotImplementedError

    def _find_knns_falconn(self, x, output):
        # Late falconn query_object construction
        # Since I suppose there might be an error
        # if table.setup() will be called after
        if self._falconn_query_object is None:
            self._falconn_query_object = self._falconn_table.construct_query_object()
            self._falconn_query_object.set_num_probes(self._FALCONN_NB_TABLES)

        missing_indices = np.zeros(output.shape, dtype=np.bool)

        for i in range(x.shape[0]):
            query_res = self._falconn_query_object.find_k_nearest_neighbors(x[i], self._NEIGHBORS)

            try:
                output[i, :] = query_res
            except:
                # mark missing indices
                missing_indices[i, len(query_res):] = True
                output[i, :len(query_res)] = query_res

        return missing_indices

    def _find_knns_faiss(self, x, output):
        neighbor_distance, neighbor_index = self._faiss_index.search(x, self._NEIGHBORS)

        missing_indices = neighbor_distance == -1
        d1 = neighbor_index.reshape(-1)

        output.reshape(-1)[np.logical_not(missing_indices.flatten())] = d1[np.logical_not(missing_indices.flatten())]

        return missing_indices


def get_activations(dataloader, model, layers):
    activations = {}
    activations['activations'] = {}
    activations['targets'] = None

    for layer_name in layers:
        print('## Fetching Activations from Layer {}'.format(layer_name))

        # Get activations for the data
        # print("from layers")
        # import pdb; pdb.set_trace()
        layer = layers[layer_name]        
        activations['activations'][layer_name], targets = get_activations_from_layer(dataloader, model, layer)

        # Get the targets of that data
        # print("targets")
        if targets is not None:
            if activations['targets'] is not None:
                np.testing.assert_array_equal(activations['targets'], targets)
            else:
                activations['targets'] = targets

        # print()

    return activations

def get_activations_from_layer(dataloader, model, layer):
    activations = []
    targets = []

    # Define hook for fetching the activations
    def hook(module, input, output):
        layer_activations = output.squeeze().detach().cpu().numpy()

        if len(layer_activations.shape) == 4:
            layer_activations = layer_activations.reshape(layer_activations.shape[0], -1)
        
        activations.append(layer_activations)

    handle = layer.register_forward_hook(hook)

    # Fetch activations
    for i, batch in enumerate(dataloader):
        get_debug_info('Processing Batch {}'.format(i))
        get_debug_info("batch len : ", len(batch))
        get_debug_info("batch size: ", batch[0].size())
        if use_cuda:
            batch = [b.cuda() for b in batch]
            # batch = batch.cuda()

        _ = model(batch[0])

        if len(batch) > 1:
          targets.append(batch[1].detach().cpu().numpy())

    get_debug_info("done!")

    # Remove hook
    handle.remove()

    # Return activations and targets
    activations = np.concatenate(activations)

    if targets:
        targets = np.hstack(targets)
    else:
        None

    return activations, targets

class DkNN:

    def __init__(self, model, nb_classes, neighbors, layers, trainloader, nearest_neighbor_backend, nb_tables=200, number_bits=17):
        """
        Implementation of the DkNN algorithm, see https://arxiv.org/abs/1803.04765 for more details
        :param model: model to be used
        :param nb_classes: the number of classes in the task
        :param neighbors: number of neighbors to find per layer
        :param layers: a list of layer names to include in the DkNN
        :param trainloader: data loader for the training data
        :param nearest_neighbor_backend: falconn or faiss to be used for LSH
        :param nb_tables: number of tables used by FALCONN to perform locality-sensitive hashing.
        :param number_bits: number of hash bits used by LSH.
        """
        print('---------- DkNN init')
        print()

        self.model = model
        self.nb_classes = nb_classes
        self.neighbors = neighbors
        self.layers = layers
        self.backend = nearest_neighbor_backend
        self.nb_tables = nb_tables
        self.number_bits = number_bits

        self.nb_cali = -1
        self.calibrated = False   

        # Compute training data activations
        activations = get_activations(trainloader, model, layers)
        self.train_activations = activations['activations']
        self.train_labels = activations['targets']

        # Build locality-sensitive hashing tables for training representations
        self.train_activations_lsh = copy.copy(self.train_activations)
        self.init_lsh()

    def init_lsh(self):
        """
        Initializes locality-sensitive hashing with FALCONN to find nearest neighbors in training data
        """
        self.query_objects = {} # contains the object that can be queried to find nearest neighbors at each layer
        self.centers = {} # mean of training data representation per layer (that needs to be substracted before NearestNeighbor)

        print("## Constructing the NearestNeighbor tables")

        for layer in self.layers:
            print("Constructing table for {}".format(layer))

            # Normalize all the lenghts, since we care about the cosine similarity
            self.train_activations_lsh[layer] /= np.linalg.norm(self.train_activations_lsh[layer], axis=1).reshape(-1, 1)

            # Center the dataset and the queries: this improves the performance of LSH quite a bit
            center = np.mean(self.train_activations_lsh[layer], axis=0)
            self.train_activations_lsh[layer] -= center
            self.centers[layer] = center

            # Constructing nearest neighbor table
            self.query_objects[layer] = NearestNeighbor(
                backend=self.backend,
                dimension=self.train_activations_lsh[layer].shape[1],
                number_bits=self.number_bits,
                neighbors=self.neighbors,
                nb_tables=self.nb_tables,
            )

            self.query_objects[layer].add(self.train_activations_lsh[layer])

        print("done!")
        print()


    def calibrate(self, calibloader):
        """
        Runs the DkNN on holdout data to calibrate the credibility metric
        :param calibloader: data loader for the calibration loader
        """
        print('---------- DkNN calibrate')
        print()

        # Compute calibration data activations
        self.nb_cali = len(calibloader.dataset)

        activations = get_activations(calibloader, self.model, self.layers)
        self.cali_activations = activations['activations']
        self.cali_labels = activations['targets']

        print("## Starting calibration of DkNN")

        cali_knns_ind, cali_knns_labels = self.find_train_knns(self.cali_activations)

        assert all([v.shape == (self.nb_cali, self.neighbors) for v in cali_knns_ind.values()])
        assert all([v.shape == (self.nb_cali, self.neighbors) for v in cali_knns_labels.values()])

        cali_knns_not_in_class = self.nonconformity(cali_knns_labels)
        cali_knns_not_in_l = np.zeros(self.nb_cali, dtype=np.int32)

        for i in range(self.nb_cali):
            cali_knns_not_in_l[i] = cali_knns_not_in_class[i, self.cali_labels[i]]

        cali_knns_not_in_l_sorted = np.sort(cali_knns_not_in_l)
        self.cali_nonconformity = np.trim_zeros(cali_knns_not_in_l_sorted, trim='f')
        self.nb_cali = self.cali_nonconformity.shape[0]
        self.calibrated = True

        print("DkNN calibration complete")

    def find_train_knns(self, data_activations):
        """
        Given a data_activation dictionary that contains a np array with activations for each layer,
        find the knns in the training data
        """
        knns_ind = {}
        knns_labels = {}

        for layer in self.layers:
            # Pre-process representations of data to normalize and remove training data mean
            data_activations_layer = copy.copy(data_activations[layer])
            nb_data = data_activations_layer.shape[0]
            data_activations_layer /= np.linalg.norm(data_activations_layer, axis=1).reshape(-1, 1)
            data_activations_layer -= self.centers[layer]

            # Use FALCONN to find indices of nearest neighbors in training data
            knns_ind[layer] = np.zeros((data_activations_layer.shape[0], self.neighbors), dtype=np.int32)
            knn_errors = 0

            knn_missing_indices = self.query_objects[layer].find_knns(data_activations_layer, knns_ind[layer])
            knn_errors += knn_missing_indices.flatten().sum()

            # Find labels of neighbors found in the training data
            knns_labels[layer] = np.zeros((nb_data, self.neighbors), dtype=np.int32)

            knns_labels[layer].reshape(-1)[
                np.logical_not(knn_missing_indices.flatten())
            ] = self.train_labels[
                knns_ind[layer].reshape(-1)[np.logical_not(knn_missing_indices.flatten())]                    
            ]

        return knns_ind, knns_labels

    def nonconformity(self, knns_labels):
        """
        Given an dictionary of nb_data x nb_classes dimension, compute the nonconformity of
        each candidate label for each data point: i.e. the number of knns whose label is
        different from the candidate label
        """
        nb_data = knns_labels[list(self.layers.keys())[0]].shape[0]
        knns_not_in_class = np.zeros((nb_data, self.nb_classes), dtype=np.int32)

        for i in range(nb_data):
            # Compute number of nearest neighbors per class
            knns_in_class = np.zeros((len(self.layers), self.nb_classes), dtype=np.int32)

            for layer_id, layer in enumerate(self.layers):
                knns_in_class[layer_id, :] = np.bincount(knns_labels[layer][i], minlength=self.nb_classes)

            # Compute number of knns in other class than class_id
            for class_id in range(self.nb_classes):
                knns_not_in_class[i, class_id] = np.sum(knns_in_class) - np.sum(knns_in_class[:, class_id])

        return knns_not_in_class

    def fprop(self, testloader):
        """
        Performs a forward pass through the DkNN on an numpy array of data
        """
        print('---------- DkNN predict')
        print()

        if not self.calibrated:
            raise ValueError("DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring")

        # Compute test data activations
        activations = get_activations(testloader, self.model, self.layers)
        data_activations = activations['activations']
        _, knns_labels = self.find_train_knns(data_activations)

        # Calculate nonconformity
        knns_not_in_class = self.nonconformity(knns_labels)
        print('Nonconformity calculated')

        # Create predictions, confidence and credibility
        _, _, creds = self.preds_conf_cred(knns_not_in_class)
        print('Predictions created')

        return creds, activations['targets']

    def preds_conf_cred(self, knns_not_in_class):
        """
        Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
        the DkNN's prediction, confidence and credibility
        """
        nb_data = knns_not_in_class.shape[0]
        preds_knn = np.zeros(nb_data, dtype=np.int32)
        confs = np.zeros((nb_data, self.nb_classes), dtype=np.float32)
        creds = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

        for i in range(nb_data):
            # p-value of test input for each class
            p_value = np.zeros(self.nb_classes, dtype=np.float32)

            for class_id in range(self.nb_classes):
                # p-value of (test point, candidate label)
                p_value[class_id] = (float(self.nb_cali) - bisect_left(self.cali_nonconformity, knns_not_in_class[i, class_id])) / float(self.nb_cali)

            preds_knn[i] = np.argmax(p_value)
            confs[i, preds_knn[i]] = 1. - np.sort(p_value)[-2]
            creds[i, preds_knn[i]] = p_value[preds_knn[i]]

        return preds_knn, confs, creds



    def get_characteristics(self, testloader):
        """
        Performs a forward pass through the DkNN on an numpy array of data
        """
        print('---------- get characteristics')
        print()

        if not self.calibrated:
            raise ValueError("DkNN needs to be calibrated by calling DkNNModel.calibrate method once before inferring")

        # Compute test data activations
        activations = get_activations(testloader, self.model, self.layers)
        data_activations = activations['activations']
        _, knns_labels = self.find_train_knns(data_activations)

        # Calculate nonconformity
        knns_not_in_class = self.nonconformity(knns_labels)
        print('Nonconformity calculated')

        # Create predictions, confidence and credibility
        p_values = self.characteristics(knns_not_in_class)
        print('Pvalues created')

        return p_values


    def characteristics(self, knns_not_in_class):
        """
        Given an array of nb_data x nb_classes dimensions, use conformal prediction to compute
        the DkNN's prediction, confidence and credibility
        """
        nb_data = knns_not_in_class.shape[0]
        p_values = np.zeros((nb_data, self.nb_classes), dtype=np.float32)

        for i in range(nb_data):
            # p-value of test input for each class
            p_value = np.zeros(self.nb_classes, dtype=np.float32)

            for class_id in range(self.nb_classes):
                # p-value of (test point, candidate label)
                p_value[class_id] = (float(self.nb_cali) - bisect_left(self.cali_nonconformity, knns_not_in_class[i, class_id])) / float(self.nb_cali)
            
            p_values[i] = p_value

        return p_values


def plot_reliability_diagram(confidence, labels):
    """
    Takes in confidence values (e.g. output of softmax or DkNN confidences) for
    predictions and correct labels for the data, plots a reliability diagram
    :param confidence: nb_samples x nb_classes with confidence scores
    :param labels: targets
    """
    assert len(confidence.shape) == 2
    assert len(labels.shape) == 1
    assert confidence.shape[0] == labels.shape[0]

    if confidence.max() <= 1.:
        # confidence array is output of softmax
        bins_start = [b / 10. for b in range(0, 10)]
        bins_end = [b / 10. for b in range(1, 11)]
        bins_center = [(b + .5) / 10. for b in range(0, 10)]
        preds_conf = np.max(confidence, axis=1)
        preds_l = np.argmax(confidence, axis=1)
    else:
        raise ValueError('Confidence values go above 1')

    print(preds_conf.shape, preds_l.shape)

    # Create var for reliability diagram (Will contain mean accuracies for each bin)
    reliability_diag = []
    num_points = []  # keeps the number of points in each bar

    # Find average accuracy per confidence bin
    for bin_start, bin_end in zip(bins_start, bins_end):
        above = preds_conf >= bin_start

        if bin_end == 1.:
            below = preds_conf <= bin_end
        else:
            below = preds_conf < bin_end

        mask = np.multiply(above, below)
        num_points.append(np.sum(mask))

        bin_mean_acc = max(0, np.mean(preds_l[mask] == labels[mask]))
        reliability_diag.append(bin_mean_acc)

    # Plot diagram
    assert len(reliability_diag) == len(bins_center)
    #print(reliability_diag)
    #print(bins_center)
    #print(num_points)

    fig, ax1 = plt.subplots()
    _ = ax1.bar(bins_center, reliability_diag, width=.1, alpha=0.8, edgecolor = "black")
    plt.xlim([0, 1.])
    ax1.set_ylim([0, 1.])

    ax2 = ax1.twinx()

    #print(sum(num_points))

    ax2.plot(bins_center, num_points, color='r', linestyle='-', linewidth=7.0)
    ax2.set_ylabel('Number of points in the data', fontsize=16, color='r')

    if len(np.argwhere(confidence[0] != 0.)) == 1:
        # This is a DkNN diagram
        ax1.set_xlabel('Prediction Credibility', fontsize=16)
    else:
        # This is a softmax diagram
        ax1.set_xlabel('Prediction Confidence', fontsize=16)

    ax1.set_ylabel('Prediction Accuracy', fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14, colors='r')
    fig.tight_layout()
    plt.show()


class CustomTensorDatasetInd(Dataset):
    """
    TensorDataset with support for transforms
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class CustomTensorDataset(Dataset):
    """
    TensorDataset with support for transforms
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        # assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def fgsm_attack(image, epsilon, data_grad, normalized=True):
    # If image is normalized, denormalize with MNIST statistics
    if normalized:
        image = image * 0.3081 + 0.1307   # MNIST statistics

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # If image should be normalized, normalize with MNIST statistics
    if normalized:
        perturbed_image = (perturbed_image - 0.1307) / 0.3081  

    # Return the perturbed image
    return perturbed_image


def get_softmax_scores(model, dataloader):
    model.eval()
    softmax_scores = []
    
    with torch.no_grad():
        for data in dataloader:
            # Get the inputs; data is a list of [inputs, labels]
            inputs, targets = data

            if use_cuda:
              inputs, targets = inputs.cuda(), targets.cuda()
            
            # Forward + loss + correct
            outputs = model(inputs)
            criterion(outputs, targets)
            
            # Get softmax scores
            softmax_output = torch.nn.functional.softmax(outputs.data, dim=1)
            pred, pred_idx = torch.max(softmax_output, 1)
            softmax_score = np.zeros((pred.shape[0],10), dtype=np.float32)

            pred = pred.detach().cpu().numpy()
            pred_idx = pred_idx.detach().cpu().numpy()

            softmax_score[np.arange(pred_idx.shape[0]), pred_idx] = pred
            softmax_scores.append(softmax_score)
    
    return np.vstack(softmax_scores)


def create_adversarials(model, dataloader, epsilon):
    adv_examples = []
    targets = []
    init_preds = []
    final_preds = []

    # Loop over all examples in data set
    for data, target in dataloader:

        # import pdb; pdb.set_trace()

        # If we run on GPU, send data and target to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # Calculate the loss
        loss = criterion(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # Save adversarial examples
        adv_ex = perturbed_data.detach().cpu().numpy()
        adv_examples.append(adv_ex)
        targets.append(target.item())
        init_preds.append(init_pred.item())
        final_preds.append(final_pred.item())

    # Return the accuracy and an adversarial example
    result = {
        'adv_examples': np.vstack(adv_examples),
        'targets': targets,
        'init_preds': np.array(init_preds),
        'final_preds': np.array(final_preds)
    }
    return result


def calculate(args, model, images, images_advs, layers, get_layer_feature_maps, activation):
    nb_classes=args.num_classes
    neighbors=nb_classes
    number_bits=17
    nb_tables=200

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    transform_ind = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # indices = torch.arange(10)
    # trainset = data_utils.Subset(trainset, indices)

    trainloader_dknn = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=False, num_workers=2
    )

    # orig_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # orig_testset_size = len(orig_testset)
    # calibset_size = 750    
    # testset_size = orig_testset_size - calibset_size

    attack = args.attack

    # import pdb; pdb.set_trace()
    if attack == 'std':
        attack = 'std/8_255'

    te_data_path = '/home/lorenzp/adversialml/src/data/attacks/run_{}/cif10/wrn_28_10_10/{}/images_te'.format(args.run_nr, attack)
    te_targets_path = '/home/lorenzp/adversialml/src/data/attacks/run_{}/cif10/wrn_28_10_10/{}/labels_te'.format(args.run_nr, attack)

    te_data    =  torch.FloatTensor(torch.stack(torch.load(te_data_path)).cpu())
    te_targets =  torch.LongTensor( torch.load(te_targets_path) ).cpu()

    # te_data    =  np.stack(torch.load(te_data_path,    map_location='cpu'))
    # te_targets =  np.stack(torch.load(te_targets_path, map_location='cpu'))

    # import pdb; pdb.set_trace()
    testset_custom = CustomTensorDataset(tensors=[te_data, te_targets], transform=transform_ind)

    calibset_size = 750    
    testset_size = len(testset_custom) - calibset_size
    testset, calibset = torch.utils.data.random_split(testset_custom, [testset_size, calibset_size])

    # testset = data_utils.Subset(testset, indices)
    # calibset = data_utils.Subset(calibset, indices)


    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_train, shuffle=False, num_workers=2
    )

    testloader_adv = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2
    )

    calibloader = torch.utils.data.DataLoader(
        calibset, batch_size=batch_size_train, shuffle=False, num_workers=2
    )

    # adversarials = create_adversarials(model, testloader_adv, 0.03)
    # adv_data = torch.FloatTensor(adversarials['adv_examples'])
    # adv_targets = torch.LongTensor(adversarials['targets'])

    adv_data_path = '/home/lorenzp/adversialml/src/data/attacks/run_{}/cif10/wrn_28_10_10/{}/images_te_adv'.format(args.run_nr, attack)
    adv_targets_path = '/home/lorenzp/adversialml/src/data/attacks/run_{}/cif10/wrn_28_10_10/{}/labels_te_adv'.format(args.run_nr, attack)

    adv_data    = torch.FloatTensor( torch.stack(torch.load(adv_data_path)).cpu())
    adv_targets = torch.LongTensor( torch.load(adv_targets_path) ).cpu()

    # adv_data    =  np.stack(torch.load(adv_data_path, map_location='cpu'))
    # adv_targets =  torch.load(adv_targets_path, map_location='cpu')

    advdataset = CustomTensorDataset(tensors=(adv_data, adv_targets), transform=transform_ind)
    advloader  = DataLoader(advdataset, batch_size=batch_size_test, shuffle=False, num_workers=2)


    print('trainset size: {}'.format(len(trainloader_dknn.dataset)))
    print('testset size:  {}'.format(len(testloader.dataset)))
    print('calibset size: {}'.format(len(calibloader.dataset)))

    dknn = DkNN(model=model, nb_classes=nb_classes, neighbors=neighbors, layers=layers, trainloader=trainloader_dknn, \
        nearest_neighbor_backend=NearestNeighbor.BACKEND.FALCONN, nb_tables=nb_tables, number_bits=number_bits)
    dknn.calibrate(calibloader)

    # path_dir = "./data/extracted_characteristics/run_1/cif10/wrn_28_10_10/fgsm/DkNN/"
    # torch.save(dknn,       os.path.join(path_dir, 'dknn'), pickle_protocol=4)

    characteristics     = dknn.get_characteristics(testloader)
    # import pdb; pdb.set_trace()
    characteristics_adv = dknn.get_characteristics(advloader)
    # import pdb; pdb.set_trace()

    dknn_preds_testset, test_targets = dknn.fprop(testloader)
    print()
    print('---------------------------------------------')
    print('preds:   {}'.format(dknn_preds_testset.shape))
    print('targets: {}'.format(test_targets.shape))   

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     plot_reliability_diagram(dknn_preds_testset, test_targets)

    # torch.save(dknn_preds_testset, os.path.join(path_dir, 'dknn_preds_testset'), pickle_protocol=4)
    # torch.save(test_targets,       os.path.join(path_dir, 'test_targets'), pickle_protocol=4)

    dknn_preds_advset, adv_targets = dknn.fprop(advloader)
    print()
    print('---------------------------------------------')
    print('preds:   {}'.format(dknn_preds_advset.shape))
    print('targets: {}'.format(test_targets.shape))


    path_dir = "./data/extracted_characteristics/run_{}/cif10/wrn_28_10_10/{}/DkNN/".format(args.run_nr, attack)
    # torch.save(dknn_preds_advset, os.path.join(path_dir, 'dknn_preds_advset'), pickle_protocol=4)
    # torch.save(adv_targets,       os.path.join(path_dir, 'adv_targets'), pickle_protocol=4)

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     plot_reliability_diagram(dknn_preds_advset, test_targets)

    softmax_scores_testset = get_softmax_scores(model, testloader); softmax_scores_testset.shape
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     plot_reliability_diagram(softmax_scores_testset, test_targets)

    torch.save(softmax_scores_testset, os.path.join(path_dir, 'softmax_scores_testset'), pickle_protocol=4)        

    softmax_scores_advset = get_softmax_scores(model, advloader); softmax_scores_advset.shape
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     plot_reliability_diagram(softmax_scores_advset, adv_targets)
    
    torch.save(softmax_scores_advset, os.path.join(path_dir, 'softmax_scores_advset'), pickle_protocol=4)
    # import pdb; pdb.set_trace()

    return characteristics, characteristics_adv


def calculate_test(args, model, images, images_advs, layers, get_layer_feature_maps, activation):
    nb_classes=args.num_classes
    neighbors=nb_classes
    number_bits=17
    nb_tables=200

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    trainloader_dknn = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=False, num_workers=2
    )

    orig_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    orig_testset_size = len(orig_testset)
    calibset_size = 750    
    testset_size = orig_testset_size - calibset_size
    testset, calibset = torch.utils.data.random_split(orig_testset, [testset_size, calibset_size])

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_train, shuffle=False, num_workers=2
    )

    testloader_adv = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=2
    )

    calibloader = torch.utils.data.DataLoader(
        calibset, batch_size=batch_size_train, shuffle=False, num_workers=2
    )

    print('trainset size: {}'.format(len(trainloader_dknn.dataset)))
    print('testset size:  {}'.format(len(testloader.dataset)))
    print('calibset size: {}'.format(len(calibloader.dataset)))

    adversarials = create_adversarials(model, testloader_adv, 0.03)

    adv_data = torch.FloatTensor(adversarials['adv_examples'])
    adv_targets = torch.LongTensor(adversarials['targets'])

    advdataset = CustomTensorDataset(tensors=(adv_data, adv_targets), transform=None)
    advloader  = DataLoader(advdataset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    dknn = DkNN(model=model, nb_classes=nb_classes, neighbors=neighbors, layers=layers, trainloader=trainloader_dknn, nearest_neighbor_backend=NearestNeighbor.BACKEND.FALCONN, nb_tables=nb_tables, number_bits=number_bits)
    dknn.calibrate(calibloader)

    path_dir = "./data/extracted_characteristics/run_1/cif10/wrn_28_10_10/fgsm/DkNN/"
    torch.save(dknn,       os.path.join(path_dir, 'dknn'), pickle_protocol=4)

    dknn_preds_testset, test_targets = dknn.fprop(testloader)
    print()
    print('---------------------------------------------')
    print('preds:   {}'.format(dknn_preds_testset.shape))
    print('targets: {}'.format(test_targets.shape))   

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plot_reliability_diagram(dknn_preds_testset, test_targets)

    torch.save(dknn_preds_testset, os.path.join(path_dir, 'dknn_preds_testset'), pickle_protocol=4)
    torch.save(test_targets,       os.path.join(path_dir, 'test_targets'), pickle_protocol=4)

    dknn_preds_advset, adv_targets = dknn.fprop(advloader)
    print()
    print('---------------------------------------------')
    print('preds:   {}'.format(dknn_preds_advset.shape))
    print('targets: {}'.format(test_targets.shape))

    # import pdb; pdb.set_trace()
    path_dir = "./data/extracted_characteristics/run_1/cif10/wrn_28_10_10/fgsm/DkNN/"
    torch.save(dknn_preds_advset, os.path.join(path_dir, 'dknn_preds_advset'), pickle_protocol=4)
    torch.save(adv_targets,       os.path.join(path_dir, 'adv_targets'), pickle_protocol=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plot_reliability_diagram(dknn_preds_advset, test_targets)

    softmax_scores_testset = get_softmax_scores(model, testloader); softmax_scores_testset.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plot_reliability_diagram(softmax_scores_testset, test_targets)

    torch.save(softmax_scores_testset, os.path.join(path_dir, 'softmax_scores_testset'), pickle_protocol=4)        

    softmax_scores_advset = get_softmax_scores(model, advloader); softmax_scores_advset.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plot_reliability_diagram(softmax_scores_advset, adv_targets)
    
    torch.save(softmax_scores_advset, os.path.join(path_dir, 'softmax_scores_advset'), pickle_protocol=4)
