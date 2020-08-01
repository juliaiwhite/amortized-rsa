"""
Train an RNN decoder to make binary predictions;
then train an RNN language model to generate sequences
"""

import contextlib
import random
from collections import defaultdict
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import models
import vision
import util
from data import ShapeWorld
from run import run
import data
from shapeworld import SHAPES, COLORS, VOCAB

def init_metrics():
    """
    Initialize the metrics for this training run. This is a defaultdict, so
    metrics not specified here can just be appended to/assigned to during
    training.
    Returns
    -------
    metrics : `collections.defaultdict`
        All training metrics
    """
    metrics = defaultdict(list)
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['best_epoch'] = 0
    return metrics

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Train', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', default='single')
    args = parser.parse_args()
    
    if args.dataset == 'single':
        language_model_data = ['./data/single/reference-1000-55.npz','./data/single/reference-1000-56.npz','./data/single/reference-1000-57.npz','./data/single/reference-1000-58.npz','./data/single/reference-1000-59.npz']
    elif args.dataset == 'chairs':
        language_model_data = ['./data/'+args.dataset+'/data_1000_0.npz','./data/'+args.dataset+'/data_1000_1.npz','./data/'+args.dataset+'/data_1000_2.npz','./data/'+args.dataset+'/data_1000_3.npz','./data/'+args.dataset+'/data_1000_4.npz','./data/'+args.dataset+'/data_1000_5.npz','./data/'+args.dataset+'/data_1000_6.npz','./data/'+args.dataset+'/data_1000_7.npz','./data/'+args.dataset+'/data_1000_8.npz','./data/'+args.dataset+'/data_1000_9.npz','./data/'+args.dataset+'/data_1000_10.npz','./data/'+args.dataset+'/data_1000_11.npz','./data/'+args.dataset+'/data_1000_12.npz','./data/'+args.dataset+'/data_1000_13.npz','./data/'+args.dataset+'/data_1000_14.npz','./data/'+args.dataset+'/data_1000_15.npz','./data/'+args.dataset+'/data_1000_16.npz','./data/'+args.dataset+'/data_1000_17.npz','./data/'+args.dataset+'/data_1000_18.npz','./data/'+args.dataset+'/data_1000_19.npz','./data/'+args.dataset+'/data_1000_20.npz','./data/'+args.dataset+'/data_1000_21.npz','./data/'+args.dataset+'/data_1000_22.npz','./data/'+args.dataset+'/data_1000_23.npz','./data/'+args.dataset+'/data_1000_24.npz','./data/'+args.dataset+'/data_1000_25.npz','./data/'+args.dataset+'/data_1000_26.npz']
    elif args.dataset == 'colors':
        language_model_data = ['./data/'+args.dataset+'/data_1000_0.npz','./data/'+args.dataset+'/data_1000_1.npz','./data/'+args.dataset+'/data_1000_2.npz','./data/'+args.dataset+'/data_1000_3.npz','./data/'+args.dataset+'/data_1000_4.npz','./data/'+args.dataset+'/data_1000_5.npz','./data/'+args.dataset+'/data_1000_6.npz','./data/'+args.dataset+'/data_1000_7.npz','./data/'+args.dataset+'/data_1000_8.npz','./data/'+args.dataset+'/data_1000_9.npz','./data/'+args.dataset+'/data_1000_10.npz','./data/'+args.dataset+'/data_1000_11.npz','./data/'+args.dataset+'/data_1000_12.npz','./data/'+args.dataset+'/data_1000_13.npz','./data/'+args.dataset+'/data_1000_14.npz','./data/'+args.dataset+'/data_1000_15.npz','./data/'+args.dataset+'/data_1000_16.npz','./data/'+args.dataset+'/data_1000_17.npz','./data/'+args.dataset+'/data_1000_18.npz','./data/'+args.dataset+'/data_1000_19.npz','./data/'+args.dataset+'/data_1000_20.npz','./data/'+args.dataset+'/data_1000_21.npz','./data/'+args.dataset+'/data_1000_22.npz','./data/'+args.dataset+'/data_1000_23.npz','./data/'+args.dataset+'/data_1000_24.npz','./data/'+args.dataset+'/data_1000_25.npz','./data/'+args.dataset+'/data_1000_26.npz','./data/'+args.dataset+'/data_1000_27.npz','./data/'+args.dataset+'/data_1000_28.npz','./data/'+args.dataset+'/data_1000_29.npz','./data/'+args.dataset+'/data_1000_30.npz','./data/'+args.dataset+'/data_1000_31.npz','./data/'+args.dataset+'/data_1000_32.npz','./data/'+args.dataset+'/data_1000_33.npz','./data/'+args.dataset+'/data_1000_34.npz','./data/'+args.dataset+'/data_1000_35.npz','./data/'+args.dataset+'/data_1000_36.npz','./data/'+args.dataset+'/data_1000_37.npz','./data/'+args.dataset+'/data_1000_38.npz','./data/'+args.dataset+'/data_1000_39.npz','./data/'+args.dataset+'/data_1000_40.npz','./data/'+args.dataset+'/data_1000_41.npz','./data/'+args.dataset+'/data_1000_42.npz','./data/'+args.dataset+'/data_1000_43.npz','./data/'+args.dataset+'/data_1000_44.npz','./data/'+args.dataset+'/data_1000_45.npz']
        
    # Vocab
    vocab = torch.load('./models/'+str(args.dataset)+'/vocab.pt')

    speaker_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)

    # Model
    language_model = models.LanguageModel(speaker_embs)
    listener = None
    if args.cuda:
        language_model = language_model.cuda()
            
    # Optimization
    optimizer = optim.Adam(list(language_model.parameters()),lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Metrics
    metrics = init_metrics()

    # Pretrain
    for epoch in range(args.epochs):
        # Train
        data_file = language_model_data[0:len(language_model_data)-1]
        train_metrics, _ = run(epoch, data_file, 'train', 'language_model', language_model, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.cuda)
        # Validation
        data_file = [language_model_data[-1]]
        val_metrics, _ = run(epoch, data_file, 'val', 'language_model', language_model, listener, optimizer, loss, vocab, args.batch_size, args.cuda, args.cuda)
        # Update metrics, prepending the split name
        for metric, value in train_metrics.items():
            metrics['train_{}'.format(metric)].append(value)
        for metric, value in val_metrics.items():
            metrics['val_{}'.format(metric)].append(value)
        metrics['current_epoch'] = epoch
        # Use validation accuracy to choose the best model
        is_best = val_metrics['acc'] > metrics['best_acc']
        if is_best:
            metrics['best_acc'] = val_metrics['acc']
            metrics['best_loss'] = val_metrics['loss']
            metrics['best_epoch'] = epoch
            best_language_model = copy.deepcopy(language_model)
        if args.debug:
            print(epoch)
            print(metrics)
        
    # Save the best model
    language_model = best_language_model
    torch.save(language_model, './models/'+str(args.dataset)+'/language-model.pt')