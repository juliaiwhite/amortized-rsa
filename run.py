import contextlib
import random
from collections import defaultdict
import copy
import math
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
    
def compute_average_metrics(meters):
    metrics = {m: vs.avg for m, vs in meters.items()}
    metrics = {
        m: v if isinstance(v, float) else v.item()
        for m, v in metrics.items()
    }
    return metrics

def _collect_outputs(meters, outputs, vocab, img, y, lang, lang_length, lis_pred, lis_scores, this_loss, this_acc, batch_size, ci_listeners, language_model, times):
    seq_prob = []
    for i,prob in enumerate(language_model.probability(lang,lang_length).cpu().numpy()):
        seq_prob.append(np.exp(prob))
        
    if ci_listeners != None:
        ci = []
        for ci_listener in ci_listeners:
            correct = (ci_listener(img,lang,lang_length).argmax(1)==y)
            acc = correct.float().mean().item()
            ci.append(acc)
            
    lang = lang.argmax(2)
    outputs['lang'].append(lang)
    outputs['pred'].append(lis_pred)
    outputs['score'].append(lis_scores)
    meters['loss'].append(this_loss.cpu().numpy())
    meters['acc'].append(this_acc)
    meters['prob'].append(seq_prob)
    if ci_listeners != None:
        meters['ci_acc'].append(ci)
    meters['length'].append(lang_length.cpu().numpy()-2)
    colors = 0
    for color in [4, 6, 9, 10, 11, 14]:
        colors += (lang == color).sum(dim=1).float()
    shapes = 0
    for shape in [7, 8, 12, 13]:
        shapes += (lang == shape).sum(dim=1).float()
    meters['colors'].append(colors.cpu().numpy())
    meters['shapes'].append(shapes.cpu().numpy())
    meters['time'].append(times)
    return meters, outputs

def _generate_utterance(token_1, token_2, batch_size, max_len):
    lang = torch.zeros(batch_size, max_len, len(vocab['w2i'].keys())).to(lang.device)
    lang[:, 0, data.SOS_IDX] = 1
    lang[:, 1, token_1] = 1
    if token_2:
        lang[:, 2, token_2] = 1
        lang[:, 3, data.EOS_IDX] = 1
        lang[:, 4:, data.PAD_IDX] = 1
        lang_length = 4*torch.ones(batch_size)
    else:
        lang[:, 2, data.EOS_IDX] = 1
        lang[:, 3:, data.PAD_IDX] = 1
        lang_length = 3*torch.ones(batch_size)
    lang = lang.unsqueeze(0)
    lang_length = lang_length.unsqueeze(0)
    return lang, lang_length

def run(data_file, split, model_type, speaker, listener, optimizer, loss, vocab, batch_size, cuda, num_samples = None, srr = True, lmbd = None, test_type = None, activation = 'gumbel', ci = True, dataset = 'shapeworld', penalty = None, tau = 1, generalization = None, debug = False):
    max_len = 40
    
    if model_type == 'sample' or model_type == 'rsa':
        if generalization == None:
            internal_listener = torch.load('./models/'+dataset+'/literal_listener_0.pt')
        else:
            internal_listener = torch.load('./models/'+dataset+'/'+generalization+'_literal_listener_0.pt')
    
    language_model = torch.load('./models/'+dataset+'/language_model.pt')

    if split == 'train':
        for param in language_model.parameters():
            param.requires_grad = False
        language_model.train()
        if model_type == 's0' or model_type == 'language_model':
            speaker.train()
        elif model_type == 'l0':
            listener.train()
        else:
            speaker.train()
            if model_type == 'amortized':
                for param in listener.parameters():
                    param.requires_grad = False
            listener.train()
        context = contextlib.suppress()
    else:
        language_model.eval()
        if model_type != 'l0' and model_type != 'oracle' and model_type != 'test':
            speaker.eval()
        if model_type != 's0' and model_type != 'language_model':
            listener.eval()
        context = torch.no_grad()  # Do not evaluate gradients for efficiency

    # Initialize outputs and average meters to keep track of the epoch's running average
    outputs = {'gt_lang':[], 'lang':[], 'score':[], 'pred':[]}
    if split == 'test':
        if ci == True:
            listener_dir = './models/shapeworld/literal_listener_'
            ci_listeners = [torch.load(listener_dir+'2.pt'), torch.load(listener_dir+'3.pt'), torch.load(listener_dir+'4.pt'), torch.load(listener_dir+'5.pt'), torch.load(listener_dir+'6.pt'), torch.load(listener_dir+'7.pt'), torch.load(listener_dir+'8.pt'), torch.load(listener_dir+'9.pt'), torch.load(listener_dir+'10.pt')]
            for ci_listener in ci_listeners:
                ci_listener.eval()
            meters = {'loss':[], 'acc':[], 'prob':[], 'ci_acc':[], 'length':[], 'colors':[], 'shapes':[], 'time':[]}
        else:
            ci_listeners = None
            meters = {'loss':[], 'acc':[], 'prob':[], 'length':[], 'colors':[], 'shapes':[], 'time':[]}
    else:
        if model_type == 's0' or model_type == 'language_model' or model_type == 'l0':
            measures = ['loss', 'acc']
        else:
            measures = ['loss', 'lm loss', 'acc', 'length']
        meters = {m: util.AverageMeter() for m in measures}

    with context:
        for file in data_file:
            d = data.load_raw_data(file)
            if split == 'test':
                dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=False)
            else:
                dataloader = DataLoader(ShapeWorld(d, vocab), batch_size=batch_size, shuffle=False)
                
            for batch_i, (img, y, lang) in enumerate(dataloader):
                batch_size = img.shape[0] 
                
                # Reformat inputs
                y = y.argmax(1) # convert from onehot
                img = img.float() # convert to float
                gt_lang = lang
                if split == 'test':
                    outputs['gt_lang'].append(gt_lang)
                if model_type == 's0' or model_type == 'l0' or model_type == 'language_model' or model_type == 'oracle':
                    max_len = 40
                    length = torch.tensor([np.count_nonzero(t) for t in lang.cpu()], dtype=np.int)
                    lang[lang>=len(vocab['w2i'].keys())] = 3
                    lang = F.one_hot(lang, num_classes = len(vocab['w2i'].keys()))
                    lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
                    for B in range(lang.shape[0]):
                        for L in range(lang.shape[1]):
                            if lang[B][L].sum() == 0:
                                lang[B][L][0] = 1
                                
                if cuda:
                    img = img.cuda()
                    y = y.cuda()
                    lang = lang.cuda()
                    if model_type == 's0' or model_type == 'l0' or model_type == 'language_model':
                        length = length.cuda()

                # Refresh the optimizer
                if split == 'train':
                    optimizer.zero_grad()

                # Forward pass
                start = time.time()
                if model_type == 'l0':
                    lis_scores = listener(img, lang, length)
                elif model_type == 's0':
                    lang_out = speaker(img, lang, length, y)
                elif model_type == 'language_model':
                    lang_out = speaker(lang, length)
                elif model_type == 'sample':
                    if num_samples == 1:
                        lang, lang_length = speaker.sample(img, y)
                    else:
                        if srr:
                            langs, lang_lengths = speaker.sample(img, y)
                        else:
                            langs, lang_lengths, eos_loss = speaker(img, y)
                        langs = langs.unsqueeze(0); lang_lengths = lang_lengths.unsqueeze(0)
                        for _ in range(num_samples-1):
                            if srr:
                                lang, lang_length = speaker.sample(img, y)
                            else:
                                lang, lang_length, eos_loss = speaker(img, y)
                            lang = lang.unsqueeze(0); lang_length = lang_length.unsqueeze(0)
                            langs = torch.cat((langs, lang), 0)
                            lang_lengths = torch.cat((lang_lengths, lang_length), 0)
                        lang = langs[:,0]
                elif model_type == 'rsa':
                    langs = 0
                    lang_lengths = 0
                    for color in [4, 6, 9, 10, 11, 14, 0]:
                        for shape in [7, 8, 12, 13, 5, 0]:
                            if color == 0:
                                if shape != 0:
                                    lang, lang_length = _generate_utterance(shape, None, batch_size, max_len)
                            elif shape == 0:
                                lang, lang_length = _generate_utterance(color, None, batch_size, max_len)
                            else:
                                lang0, lang_length0 = _generate_utterance(color, shape, batch_size, max_len)
                                lang1, lang_length1 = _generate_utterance(shape, color, batch_size, max_len)       
                                lang = torch.cat((lang0.unsqueeze(0), lang1.unsqueeze(0)), 0)
                                lang_length = torch.cat((lang_length0.unsqueeze(0), lang_length1.unsqueeze(0)), 0)
                            try:
                                langs = torch.cat((langs, lang), 0)
                                lang_lengths = torch.cat((lang_lengths, lang_length), 0)
                            except:
                                langs = lang
                                lang_lengths = lang_length
                elif model_type == 'amortized':
                    if penalty == None:
                        lang, lang_length, eos_loss, lang_prob = speaker(img, y, activation = activation, tau = tau, length_penalty = False)
                    else:
                        lang, lang_length, eos_loss, lang_prob = speaker(img, y, activation = activation, tau = tau, length_penalty = True)
                elif model_type == 'test':
                    langs = torch.zeros(batch_size, max_len, len(vocab['w2i'].keys()))
                    for i in range(len(lang)):
                        color = lang[i,1]
                        shape = lang[i,2]
                        if test_type == 'color':
                            langs, lang_lengths = _generate_utterance(color, None, batch_size, max_len)
                        elif test_type == 'shape':
                            langs, lang_lengths = _generate_utterance(shape, None, batch_size, max_len)
                        elif test_type == 'color-shape':
                            langs, lang_lengths = _generate_utterance(color, shape, batch_size, max_len)
                        else:
                            langs, lang_lengths = _generate_utterance(shape, color, batch_size, max_len)
                    langs = langs.unsqueeze(0)
                    lang_lengths = lang_lengths.unsqueeze(0)
                elif model_type == 'oracle':
                    lang_length = length
                else:
                    lang, lang_length, eos_loss, lang_prob = speaker(img, y, activation = activation, tau = tau)

                # Evaluate loss and accuracy
                if model_type == 'l0':
                    this_loss = loss(lis_scores, y)
                    lis_pred = lis_scores.argmax(1)
                    this_acc = (lis_pred == y).float().mean().item()
                    
                    if split == 'train':
                        # SGD step
                        this_loss.backward()
                        optimizer.step()
                       
                    meters['loss'].update(this_loss, batch_size)
                    meters['acc'].update(this_acc, batch_size)
                elif model_type == 's0' or model_type == 'language_model':
                    lang_out = lang_out[:, :-1].contiguous()
                    lang = lang[:, 1:].contiguous()
                    lang_out = lang_out.view(batch_size*lang_out.size(1), len(vocab['w2i'].keys()))
                    lang = lang.long().view(batch_size*lang.size(1), len(vocab['w2i'].keys()))
                    this_loss = loss(lang_out.cuda(), torch.max(lang, 1)[1].cuda())

                    if split == 'train':
                        # SGD step
                        this_loss.backward()
                        optimizer.step()
                        
                    this_acc = (lang_out.argmax(1)==lang.argmax(1)).float().mean().item()

                    meters['loss'].update(this_loss, batch_size)
                    meters['acc'].update(this_acc, batch_size)
                else:
                    if model_type == 'sample' or model_type == 'rsa' or model_type == 'test':
                        if not (model_type == 'sample' and num_samples == 1):
                            if model_type == 'sample':
                                alpha = 1
                            elif model_type == 'rsa':
                                alpha = 0.0001
                            else:
                                alpha = 0
                            if model_type != 'test':
                                best_score_diff = -math.inf*torch.ones(batch_size)
                                best_lang = torch.zeros((langs.shape[1],langs.shape[2],langs.shape[3]))
                                best_lang_length = torch.zeros(lang_lengths.shape[1])
                                for lang, lang_length in zip(langs, lang_lengths):
                                    lis_scores = internal_listener(img, lang, lang_length)
                                    score_diff = ((lis_scores[:,0].cpu()-np.delete(lis_scores.cpu(), y.cpu(), axis=1).mean(axis=1)))
                                    for game in range(batch_size):
                                        score_diff[game] = (score_diff[game]-alpha*lang_length[game]).cpu()
                                        if score_diff[game]>best_score_diff[game]:
                                            best_score_diff[game] = score_diff[game]
                                            best_lang[game] = lang[game]
                                            best_lang_length[game] = lang_length[game]

                                lang = best_lang
                                lang_length = best_lang_length
                            else:
                                lang = langs.squeeze()
                                lang_length = lang_lengths.squeeze()
                        end = time.time()
                        lis_scores = listener(img, lang, lang_length)
                        
                        # Evaluate loss and accuracy
                        lis_pred = lis_scores.argmax(1)
                        correct = (lis_pred == y)
                        this_acc = correct.float().mean().item()
                        this_loss = loss(lis_scores.cuda(), y.long())
                        this_acc = correct.float().mean().item()
                                
                        if split == 'train':
                            # SGD step
                            this_loss.backward()
                            optimizer.step()
                        if split == 'test':
                            meters, outputs = _collect_outputs(meters, outputs, vocab, img, y, lang, lang_length, lis_pred, lis_scores, this_loss, this_acc, batch_size, ci_listeners, language_model, (end-start))
                        else:
                            meters['loss'].update(this_loss, batch_size)
                            meters['acc'].update(this_acc, batch_size)
                    else:
                        if split == 'train' and model_type == 'amortized' and activation == 'multinomial': # Reinforce
                            end = time.time()
                            lis_scores = listener(img, lang, lang_length, average=False)
                        elif split == 'train' and model_type == 'amortized' and activation != 'gumbel' and activation != None:
                            end = time.time()
                            lis_scores = listener(img, lang, lang_length, average=True)
                        else:
                            lang_onehot = lang.argmax(2)
                            if activation != 'gumbel' and activation != None:
                                lang = F.one_hot(lang_onehot, num_classes = len(vocab['w2i'].keys())).cuda().float()
                            lang_length = []
                            for seq in lang_onehot: 
                                lang_length.append(np.where(seq.cpu()==data.EOS_IDX)[0][0]+1)
                            lang_length = torch.tensor(lang_length).cuda()
                            end = time.time()
                            lis_scores = listener(img, lang, lang_length)
                            
                        # Evaluate loss and accuracy
                        if model_type == 'l0':
                            this_loss = loss(lis_scores, y.long())
                        elif model_type == 'amortized':
                            if activation == 'multinomial':
                                # Compute policy loss - sample from listener
                                # DETERMINISTIC REWARD
                                #  lis_choices = lis_scores.argmax(1)  # deterministically choose best
                                # STOCHASTIC REWARD
                                lis_choices = torch.distributions.Categorical(probs=lis_scores).sample()
                                returns = lis_choices == y
                                # No reward for saying nothing
                                not_zero = lang_length > 2
                                returns = (returns & not_zero).float()
                                # Length penalty - less returns
                                # In the amortized model, we don't penalize
                                # SOS, so the true length that we penalize
                                # is lang_length - 1.
                                LENGTH_PENALTY = 0.01
                                returns = returns - LENGTH_PENALTY * (lang_length.to(returns.device).float() - 1)
                                returns = torch.clamp(returns, 0.0, 1.0)
                                # Slight negative reward for getting things wrong
                                # (TODO: tweak this)
                                #  returns = (1 * returns) + (-0.1 * (1 - returns))
                                # FIXME: Should we normalize in the binary case?
                                policy_loss = (-lang_prob * returns).mean()
                                this_loss = policy_loss
                            else:
                                this_loss = loss(lis_scores,y.long())
                            this_loss = this_loss + eos_loss * float(lmbd)
                        else:
                            this_loss = loss(lis_scores, y.long())
                            
                        lis_pred = lis_scores.argmax(1)
                        correct = (lis_pred == y)
                        this_acc = correct.float().mean().item()
                        
                        if split == 'train':
                            # SGD step
                            this_loss.backward()
                            optimizer.step()
                        
                        if split == 'test':
                            meters, outputs = _collect_outputs(meters, outputs, vocab, img, y, lang, lang_length, lis_pred, lis_scores, this_loss, this_acc, batch_size, ci_listeners, language_model, (end-start))
                        else:
                            meters['loss'].update(this_loss-eos_loss*float(lmbd), batch_size)
                            meters['lm loss'].update(eos_loss*float(lmbd), batch_size)
                            meters['acc'].update(this_acc, batch_size)
                            meters['length'].update(lang_length.float().mean(), batch_size)
        
    if split == 'test':
        meters['loss'] = np.array(meters['loss']).tolist()
        meters['prob'] = [prob for sublist in meters['prob'] for prob in sublist]
        meters['length'] = [length for sublist in meters['length'] for length in sublist]
        meters['colors'] = [color for sublist in meters['colors'] for color in sublist]
        meters['shapes'] = [shape for sublist in meters['shapes'] for shape in sublist]
        metrics = meters
    else:
        metrics = compute_average_metrics(meters)
    if model_type == 'amortized':
        seq = []
        for word_index in lang.argmax(2)[0,:].cpu().numpy():
            try:
                seq.append(vocab['i2w'][word_index])
            except:
                seq.append('<UNK>')
        if debug:
            print('Generated utterance: '+' '.join(seq))
        seq = []
        for word_index in gt_lang[0,:].cpu().numpy():
            try:
                seq.append(vocab['i2w'][word_index])
            except:
                seq.append('<UNK>')
        if debug:
            print('Ground truth utterance: '+' '.join(seq))
    return metrics, outputs