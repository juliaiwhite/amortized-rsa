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
import pandas as pd

import vision
import util
from data import ShapeWorld
import data

from colors import ColorsInContext

from run import run
import models

def init_metrics():
    metrics = defaultdict(list)
    metrics['best_acc'] = 0.0
    metrics['best_loss'] = float('inf')
    metrics['best_epoch'] = 0
    return metrics

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description='Train', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dataset', default='shapeworld', help='(shapeworld or colors)')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--vocab', action='store_true', help='Generate new vocab file')
    parser.add_argument('--s0', action='store_true', help='Train literal speaker')
    parser.add_argument('--l0', action='store_true', help='Train literal listener')
    parser.add_argument('--amortized', action='store_true', help='Train amortized speaker')
    parser.add_argument('--activation', default=None)
    parser.add_argument('--penalty', default=None, help='Cost function (length)')
    parser.add_argument('--lmbd', default=0.01, help='Cost function parameter')
    parser.add_argument('--tau', default=1, type=float, help='Softmax temperature')
    parser.add_argument('--save', default='metrics.csv', help='Where to save metrics')
    parser.add_argument('--debug', action='store_true', help='Print metrics on every epoch')
    parser.add_argument('--generalization', default=None)
    args = parser.parse_args()
    
    if args.l0 and args.lr == None:
        args.lr = 0.0001
    elif args.lr == None:
        args.lr = 0.001
    
    # Data
    if args.dataset == 'shapeworld':
        if args.generalization == None:
            data_dir = './data/shapeworld/reference-1000-'
            pretrain_data = [[data_dir+'0.npz', data_dir+'1.npz', data_dir+'2.npz', data_dir+'3.npz', data_dir+'4.npz'],[data_dir+'5.npz', data_dir+'6.npz', data_dir+'7.npz', data_dir+'8.npz', data_dir+'9.npz'],[data_dir+'10.npz', data_dir+'11.npz', data_dir+'12.npz', data_dir+'13.npz', data_dir+'14.npz'],[data_dir+'15.npz', data_dir+'16.npz', data_dir+'17.npz', data_dir+'18.npz', data_dir+'19.npz'],[data_dir+'20.npz', data_dir+'21.npz', data_dir+'22.npz', data_dir+'23.npz', data_dir+'24.npz'],[data_dir+'25.npz', data_dir+'26.npz', data_dir+'27.npz', data_dir+'28.npz', data_dir+'29.npz'],[data_dir+'30.npz', data_dir+'31.npz', data_dir+'32.npz', data_dir+'33.npz', data_dir+'34.npz'],[data_dir+'35.npz', data_dir+'36.npz', data_dir+'37.npz', data_dir+'38.npz', data_dir+'39.npz'],[data_dir+'40.npz', data_dir+'41.npz', data_dir+'42.npz', data_dir+'43.npz', data_dir+'44.npz'],[data_dir+'45.npz', data_dir+'46.npz', data_dir+'47.npz', data_dir+'48.npz', data_dir+'49.npz'],[data_dir+'50.npz', data_dir+'51.npz', data_dir+'52.npz', data_dir+'53.npz', data_dir+'54.npz'],[data_dir+'70.npz', data_dir+'71.npz', data_dir+'72.npz', data_dir+'73.npz', data_dir+'74.npz']]
        else:
            data_dir = './data/shapeworld/generalization/'+args.generalization+'/reference-1000-'
            pretrain_data = [[data_dir+'0.npz', data_dir+'1.npz', data_dir+'2.npz', data_dir+'3.npz', data_dir+'4.npz'],[data_dir+'5.npz', data_dir+'6.npz', data_dir+'7.npz', data_dir+'8.npz', data_dir+'9.npz']]
        train_data = [data_dir+'60.npz', data_dir+'61.npz', data_dir+'62.npz', data_dir+'63.npz', data_dir+'64.npz']
        val_data = [data_dir+'65.npz', data_dir+'66.npz', data_dir+'67.npz', data_dir+'68.npz', data_dir+'69.npz']
        
    elif args.dataset == 'colors':
        DatasetClass = ColorsInContext
        data_dir = './data/colors/data_1000_'
        pretrain_data = [[data_dir+'0.npz', data_dir+'1.npz', data_dir+'2.npz', data_dir+'3.npz', data_dir+'4.npz', data_dir+'5.npz', data_dir+'6.npz', data_dir+'7.npz', data_dir+'8.npz', data_dir+'9.npz', data_dir+'10.npz', data_dir+'11.npz', data_dir+'12.npz', data_dir+'13.npz', data_dir+'14.npz'],[data_dir+'15.npz', data_dir+'16.npz', data_dir+'17.npz', data_dir+'18.npz', data_dir+'19.npz', data_dir+'20.npz', data_dir+'21.npz', data_dir+'22.npz', data_dir+'23.npz', data_dir+'24.npz', data_dir+'25.npz', data_dir+'26.npz', data_dir+'27.npz', data_dir+'28.npz', data_dir+'29.npz'],[data_dir+'30.npz', data_dir+'31.npz', data_dir+'32.npz', data_dir+'33.npz', data_dir+'34.npz', data_dir+'35.npz', data_dir+'36.npz', data_dir+'37.npz', data_dir+'38.npz', data_dir+'39.npz', data_dir+'40.npz', data_dir+'41.npz', data_dir+'42.npz', data_dir+'43.npz', data_dir+'44.npz']]
        train_data = [data_dir+'0.npz', data_dir+'1.npz', data_dir+'2.npz', data_dir+'3.npz', data_dir+'4.npz', data_dir+'5.npz', data_dir+'6.npz', data_dir+'7.npz', data_dir+'8.npz', data_dir+'9.npz', data_dir+'10.npz', data_dir+'11.npz', data_dir+'12.npz', data_dir+'13.npz', data_dir+'14.npz']
        val_data = [data_dir+'15.npz', data_dir+'16.npz', data_dir+'17.npz', data_dir+'18.npz', data_dir+'19.npz', data_dir+'20.npz', data_dir+'21.npz', data_dir+'22.npz', data_dir+'23.npz', data_dir+'24.npz', data_dir+'25.npz', data_dir+'26.npz', data_dir+'27.npz', data_dir+'28.npz', data_dir+'29.npz']
    else:
        raise Exception('Dataset '+args.dataset+' is not defined.')
        
    # Load or Generate Vocab
    if args.vocab:
        langs = np.array([])
        for files in pretrain_data:
            for file in files:
                d = data.load_raw_data(file)
                langs = np.append(langs, d['langs'])
        vocab = data.init_vocab(langs)
        torch.save(vocab,'./models/'+args.dataset+'/vocab.pt')
    else:
        vocab = torch.load('./models/'+args.dataset+'/vocab.pt')
    
    # Initialize Speaker and Listener Model
    speaker_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    speaker_vision = vision.Conv4()
    if args.s0:
        speaker = models.LiteralSpeaker(speaker_vision, speaker_embs)
    else:
        speaker = models.Speaker(speaker_vision, speaker_embs)
    listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
    listener_vision = vision.Conv4()
    listener = models.Listener(listener_vision, listener_embs)
    if args.cuda:
        speaker = speaker.cuda()
        listener = listener.cuda()
        
    # Optimization
    optimizer = optim.Adam(list(speaker.parameters())+list(listener.parameters()),lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Initialize Metrics
    metrics = init_metrics()
    all_metrics = []
        
    # Optimization
    optimizer = optim.Adam(list(speaker.parameters())+list(listener.parameters()),lr=args.lr)
    loss = nn.CrossEntropyLoss()

    # Initialize Metrics
    metrics = init_metrics()
    all_metrics = []

    # Pretrain Literal Listener
    if args.l0:
        if args.generalization:
            output_dir = './models/shapeworld/generalization/'+args.generalization+'/literal_listener_'
            output_files = [output_dir+'0.pt', output_dir+'1.pt']
        else:
            output_dir = './models/'+args.dataset+'/literal_listener_'
            output_files = [output_dir+'0.pt', output_dir+'1.pt', output_dir+'2.pt', output_dir+'3.pt', output_dir+'4.pt', output_dir+'5.pt', output_dir+'6.pt', output_dir+'7.pt', output_dir+'8.pt', output_dir+'9.pt', output_dir+'10.pt']
             
        for file, output_file in zip(pretrain_data,output_files):
            # Reinitialize metrics, listener model, and optimizer
            metrics = init_metrics()
            all_metrics = []
            listener_embs = nn.Embedding(len(vocab['w2i'].keys()), 50)
            listener_vision = vision.Conv4()
            listener = models.Listener(listener_vision, listener_embs)
            if args.cuda:
                listener = listener.cuda()
            optimizer = optim.Adam(list(listener.parameters()),lr=args.lr)
        
            for epoch in range(args.epochs):
                # Train one epoch
                data_file = file[0:len(file)-1]
                train_metrics, _ = run(data_file, 'train', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug = args.debug)

                # Validate
                data_file = [file[-1]]
                val_metrics, _ = run(data_file, 'val', 'l0', None, listener, optimizer, loss, vocab, args.batch_size, args.cuda, debug = args.debug)

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
                    best_listener = copy.deepcopy(listener)
                    
                if args.debug:
                    print(metrics)

            # Save the best model
            literal_listener = best_listener
            torch.save(literal_listener, output_file)
            
    # Load Literal Listener
    if args.amortized or args.s0:
        if args.generalization:
            literal_listener = torch.load('./models/shapeworld/generalization/'+args.generalization+'/literal_listener_0.pt')
            literal_listener_val = torch.load('./models/shapeworld/generalization/'+args.generalization+'/literal_listener_1.pt')
        else:
            literal_listener = torch.load('./models/'+args.dataset+'/literal_listener_0.pt')
            literal_listener_val = torch.load('./models/'+args.dataset+'/literal_listener_1.pt')

    # Train Literal Speaker
    if args.s0:
        for epoch in range(args.epochs):
            # Train one epoch
            train_metrics, _ = run(train_data, 'train', 's0', speaker, literal_listener, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug)
            
            # Validate
            val_metrics, _ = run(val_data, 'val', 's0', speaker, literal_listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, debug = args.debug)
            
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
                best_speaker = copy.deepcopy(speaker)

            if args.debug:
                print(metrics)

            # Store metrics
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)

        # Save the best model
        if args.generalization:
            torch.save(best_speaker, './models/shapeworld/generalization/'+args.generalization+'/literal_speaker.pt')
        else:
            torch.save(best_speaker, './models/'+args.dataset+'/literal_speaker.pt')
    
    # Train Amortized Speaker
    if args.amortized:
        for epoch in range(args.epochs):
            # Train one epoch
            train_metrics, _ = run(train_data, 'train', 'amortized', speaker, literal_listener, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau, debug = args.debug)
            
            # Validate
            val_metrics, _ = run(val_data, 'val', 'amortized', speaker, literal_listener_val, optimizer, loss, vocab, args.batch_size, args.cuda, lmbd = args.lmbd, activation = args.activation, dataset = args.dataset, penalty = args.penalty, tau = args.tau, debug = args.debug)

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
                best_speaker = copy.deepcopy(speaker)

            if args.debug:
                print(metrics)

            # Store metrics
            metrics_last = {k: v[-1] if isinstance(v, list) else v
                            for k, v in metrics.items()}
            all_metrics.append(metrics_last)
            pd.DataFrame(all_metrics).to_csv(args.save, index=False)

        # Save the best model
        try:
            if args.generalization:
                if args.activation == 'multinomial':
                    torch.save(best_speaker, './models/shapeworld/generalization/'+args.generalization+'/reinforce_speaker.pt')
                else:
                    torch.save(best_speaker, './models/shapeworld/generalization/'+args.generalization+'/literal_speaker.pt')
            else:
                if args.activation == 'multinomial':
                    torch.save(best_speaker, './models/'+args.dataset+'/reinforce_speaker.pt')
                elif args.penalty == 'length':
                    torch.save(best_speaker, './models/'+args.dataset+'/amortized_speaker.pt')
        except:
            random_file = str(np.random.randint(0,1000))
            print('failed saving, now saving at '+random_file+'.pt')
            torch.save(best_speaker, './models/'+random_file+'.pt')