import contextlib
import random
from collections import defaultdict
import copy
import time

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
import data
from shapeworld import SHAPES, COLORS
from run import run
from train import init_metrics

get_inputs = False

vocab = torch.load('./models/single/vocab.pt')
print(vocab)

# Optimization
optimizer = None
loss = nn.CrossEntropyLoss()

# Metrics
metrics = init_metrics()
   
batch_size = 100
files = ['./data/single/random/reference-1000.npz','./data/single/both_needed/reference-1000.npz', './data/single/either_ok/reference-1000.npz','./data/single/shape_needed/reference-1000.npz','./data/single/color_needed/reference-1000.npz']
output_files = ['./output/single/random/','./output/single/both_needed/','./output/single/either_ok/','./output/single/shape_needed/','./output/single/color_needed/']
epoch = 0
listener_names = ['train','val','test']
listeners = ['./models/single/literal_listener_0.pt','./models/single/literal_listener_1.pt','./models/single/literal_listener_2.pt']
models = ['naive','contextual','rsa','srr','reinforce','amortized']
speakers = ['./models/single/literal_speaker.pt','./models/single/contextual_speaker.pt',
            './models/single/literal_speaker.pt','./models/single/literal_speaker.pt',
            './models/single/reinforce_speaker.pt','./models/single/amortized_speaker.pt']
run_types = ['sample','sample','rsa','sample','pragmatic','pragmatic']
activations = ['gumbel','gumbel','gumbel','gumbel','multinomial','gumbel']
num_samples = [1,1,1,5,1,1]

for listener, listener_name in zip(listeners, listener_names):
    listener = torch.load(listener)
    for model, speaker, run_type, activation, n in zip(models,speakers,run_types,activations,num_samples):
        speaker = torch.load(speaker)
        for (file, output_file) in zip(files,output_files):
            metrics, _ = run(epoch, [file], 'test', run_type, speaker, listener, optimizer, loss, vocab, batch_size, True, num_samples = n, activation = activation)
            np.save(output_file+str(model)+'_'+str(listener_name)+'_metrics.npy', metrics)