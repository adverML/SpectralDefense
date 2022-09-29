#!/usr/bin/env python3

import torch

from conf import settings

from utils import (
    get_debug_info
)

def generate_data_labels(logger, args, model, loader, wanted_samples, output_path_dir, option=2):

    clean_dataset = []
    correct = 0
    total = 0
    i = 0
    acc = ((wanted_samples, 0, 0))

    logger.log('INFO: Classify images...')
    with torch.no_grad():
        for images, labels in loader:
            if i == 0:
                logger.log( "INFO: tensor size: " + str(images.size()) )

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()
            # import pdb; pdb.set_trace()
            
            if (predicted == labels):
                clean_dataset.append([images.cpu(), labels.cpu()])

            i = i + 1
            if i % 500 == 0:
                acc  = (wanted_samples, i, 100 * correct / total)
                logger.log('INFO: Accuracy of the network on the %d test images: %d, %d %%' % acc)

            if len(clean_dataset) >= wanted_samples:
                break
    
    if option == 2: 
        logger.log("INFO: initial accuracy: {:.2f}".format(acc[-1]))
    elif option == 1: 
        logger.log("INFO: initial te accuracy: {:.2f}".format(acc[-1]))
    elif option == 0: 
        logger.log("INFO: initial tr accuracy: {:.2f}".format(acc[-1]))
    else:
        logger.log("err: logger not found!")
    
    logger.log("INFO: output_path_dir: " + output_path_dir + ", len(clean_dataset) " + str(len(clean_dataset)) )
    # print( "len labels ", len(clean_labels))
    
    return clean_dataset