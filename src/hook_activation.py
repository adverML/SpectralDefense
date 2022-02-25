import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as tmodels
from functools import partial
import collections

import pdb

# dummy data: 10 batches of images with batch size 16
dataset = [torch.rand(16,3,224,224).cuda() for _ in range(10)]

# network: a resnet50
pretrained_model = tmodels.resnet50(pretrained=True).cuda()

num_ftrs = pretrained_model.fc.in_features
print(pretrained_model)

conv_layers = []
model_weights = []
model_children = list(pretrained_model.children())
counter = 0

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.ReLU:
                    print('--------------------------------------------------')
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)


pdb.set_trace()


# # a dictionary that keeps saving the activations as they come
# activations = collections.defaultdict(list)
# def save_activation(name, mod, inp, out):
# 	activations[name].append(out.cpu())

# # Registering hooks for all the Conv2d layers
# # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
# # called repeatedly at different stages of the forward pass (like RELUs), this will save different
# # activations. Editing the forward pass code to save activations is the way to go for these cases.
# for name, m in net.named_modules():
# 	if type(m)==nn.Conv2d:
# 		# partial to assign the layer name to each hook
# 		m.register_forward_hook(partial(save_activation, name))

# # forward pass through the full dataset
# for batch in dataset:
# 	out = net(batch)

# # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
# activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

# # just print out the sizes of the saved activations as a sanity check
# for k,v in activations.items():
# 	print (k, v.size())