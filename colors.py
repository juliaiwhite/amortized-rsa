import os
import math
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize

import torch
import torch.utils.data as data
from torchvision import transforms

import shutil
from collections import Counter, OrderedDict

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)
    
class ColorsInContext(data.Dataset):

    def __init__(
            self,
            data_dir, 
            data_size = None,
            image_size = 64,
            vocab = None, 
            split = 'train', 
            context_condition = 'all',
            train_frac = 0.64,
            val_frac = 0.16,
            image_transform = None,
            min_token_occ = 2,
            max_sent_len = 16,
            random_seed = 42,
            **kwargs
        ):

        super().__init__()
        assert context_condition in ['all', 'far', 'close']

        if image_size is None:
            image_size = 64

        if data_size is not None:
            assert data_size > 0
            assert data_size <= 1

        self.data_dir = data_dir
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        self.data_size = data_size
        self.image_size = image_size
        self.vocab = vocab
        self.split = split
        self.context_condition = context_condition
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.min_token_occ = min_token_occ
        self.max_sent_len = max_sent_len
        self.random_seed = random_seed
        self.subset_indices = None

        print('1')
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = image_transform

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        cache_clean_data = os.path.join(
            self.cache_dir,
            f'clean_data_{self.context_condition}.pickle',
        )
        
        if not os.path.isfile(cache_clean_data):
            csv_path = os.path.join(self.data_dir, 'filteredCorpus.csv')
            df = pd.read_csv(csv_path)
            df = df[df['outcome'] == True]
            df = df[df['role'] == 'speaker']
            df = df.dropna()

            if self.context_condition != 'all':
                df = df[df['condition'] == self.context_condition]
            
            data = df[[
                    'clickColH',
                    'clickColS',
                    'clickColL',
                    'alt1ColH',
                    'alt1ColS',
                    'alt1ColL',
                    'alt2ColH',
                    'alt2ColS',
                    'alt2ColL',
                    'contents',
            ]]
            data = np.asarray(data)
            
            print('Saving cleaned data to pickle.')
            with open(cache_clean_data, 'wb') as fp:
                pickle.dump(data, fp)
        else:
            print('Loading cleaned data from pickle.')
            with open(cache_clean_data, 'rb') as fp:
                data = pickle.load(fp)

        data = self._process_splits(data)

        if data_size is not None:
            rs = np.random.RandomState(self.random_seed)
            n_train_total = len(data)
            indices = np.arange(n_train_total)
            n_train_total = int(math.ceil(data_size * n_train_total))
            indices = rs.choice(indices, size=n_train_total)
            data = data[indices]

            self.subset_indices = indices

        text = [d[-1] for d in data]

        if vocab is None:
            print('Building vocabulary')
            self.vocab = self.build_vocab(text)
        else:
            self.vocab = vocab

        self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']
        self.vocab_size = len(self.w2i)

        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN

        self.sos_index = self.w2i[self.sos_token]
        self.eos_index = self.w2i[self.eos_token]
        self.pad_index = self.w2i[self.pad_token]
        self.unk_index = self.w2i[self.unk_token]

        text_seq, text_len, text_raw = self._process_text(text)

        self.data = data
        self.text_seq = text_seq
        self.text_len = text_len
        self.text_raw = text_raw
        self.text = text

    def get_human_accuracy(self):
        csv_path = os.path.join(self.data_dir, 'filteredCorpus.csv')
        df = pd.read_csv(csv_path)
        df = df[df['role'] == 'listener']

        # measure using the context the user chose
        if self.context_condition != 'all':
            df = df[df['condition'] == self.context_condition]

        df = df.dropna()

        accuracy = np.asarray(df['outcome']).astype(np.float).mean()
        return accuracy

    def _process_splits(self, data):
        rs = np.random.RandomState(self.random_seed)
        rs.shuffle(data)

        n_train = int(self.train_frac * len(data))
        n_val = int((self.train_frac + self.val_frac) * len(data))
        
        if self.split == 'train':
            data = data[:n_train]
        elif self.split == 'val':
            data = data[n_train:n_val]
        elif self.split == 'test':
            data = data[n_val:]
        else:
            raise Exception(f'split {self.split} not supported.')
        return data
    
    def build_vocab(self, texts):
        w2i = dict()
        i2w = dict()
        w2c = OrderedCounter()
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        pbar = tqdm(total=len(texts))
        for text in texts:
            tokens = word_tokenize(text.lower())
            tokens = clean_tokens(tokens)
            w2c.update(tokens)
            pbar.update()
        pbar.close()

        for w, c in w2c.items():
            if c >= self.min_token_occ:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)

        return vocab

    def _process_text(self, text):
        text_seq, text_len, raw_tokens = [], [], []

        for i in range(len(text)):
            _tokens = word_tokenize(text[i].lower())
            _tokens = clean_tokens(_tokens)
            
            tokens = [SOS_TOKEN] + _tokens[:self.max_sent_len] + [EOS_TOKEN]
            length = len(tokens)
            tokens.extend([PAD_TOKEN] * (self.max_sent_len + 2 - length))
            tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in tokens]
            
            text_seq.append(tokens)
            text_len.append(length)
            raw_tokens.append(_tokens)

        text_seq = np.array(text_seq)
        text_len = np.array(text_len)

        return text_seq, text_len, raw_tokens

    def __len__(self):
        return len(self.data)
    
    def __gettext__(self, index):
        return self.text_raw[index]
    
    def __getitem__(self, index):
        h_tgt, s_tgt, l_tgt, h_alt1, s_alt1, l_alt1, h_alt2, s_alt2, l_alt2, _ = self.data[index]
       
        r_tgt, g_tgt, b_tgt = hsl2rgb(h_tgt, s_tgt / 100., l_tgt / 100.)
        r_alt1, g_alt1, b_alt1 = hsl2rgb(h_alt1, s_alt1 / 100., l_alt1 / 100.)
        r_alt2, g_alt2, b_alt2 = hsl2rgb(h_alt2, s_alt2 / 100., l_alt2 / 100.)

        color_tgt = np.array([r_tgt, g_tgt, b_tgt])[:, np.newaxis]\
            .repeat(self.image_size**2, 1).reshape(3, self.image_size, self.image_size)
        color_alt1 = np.array([r_alt1, g_alt1, b_alt1])[:, np.newaxis]\
            .repeat(self.image_size**2, 1).reshape(3, self.image_size, self.image_size)
        color_alt2 = np.array([r_alt2, g_alt2, b_alt2])[:, np.newaxis]\
            .repeat(self.image_size**2, 1).reshape(3, self.image_size, self.image_size)
        
        label = 0

        color_tgt = color_tgt.transpose(1, 2, 0).astype('uint8')
        color_alt1 = color_alt1.transpose(1, 2, 0).astype('uint8')
        color_alt2 = color_alt2.transpose(1, 2, 0).astype('uint8')

        color_tgt_pil = Image.fromarray(color_tgt).convert('RGB')
        color_alt1_pil = Image.fromarray(color_alt1).convert('RGB')
        color_alt2_pil = Image.fromarray(color_alt2).convert('RGB')

        color_tgt_pt = self.image_transform(color_tgt_pil)
        color_alt1_pt = self.image_transform(color_alt1_pil)
        color_alt2_pt = self.image_transform(color_alt2_pil)

        text_seq = torch.from_numpy(self.text_seq[index]).long()
        text_len = self.text_len[index]

        # return index, color_tgt_pt, color_alt1_pt, color_alt2_pt, text_seq, text_len, label
        return torch.cat((color_tgt_pt.unsqueeze(0), color_alt1_pt.unsqueeze(0), color_alt2_pt.unsqueeze(0)),0), label, text_seq


def hsl2rgb(H, S, L):
    assert (0 <= H <= 360) and (0 <= S <= 1) and (0 <= L <= 1)

    C = (1 - abs(2 * L - 1)) * S
    X = C * (1 - abs((H / 60.) % 2 - 1))
    m = L - C / 2.

    if H < 60:
        Rp, Gp, Bp = C, X, 0
    elif H < 120:
        Rp, Gp, Bp = X, C, 0
    elif H < 180:
        Rp, Gp, Bp = 0, C, X
    elif H < 240:
        Rp, Gp, Bp = 0, X, C
    elif H < 300:
        Rp, Gp, Bp = X, 0, C
    elif H < 360:
        Rp, Gp, Bp = C, 0, X

    R = int((Rp + m) * 255.)
    G = int((Gp + m) * 255.)
    B = int((Bp + m) * 255.)
    return (R, G, B)


def clean_tokens(tokens):
    i = 0
    while i < len(tokens):
        while (tokens[i] != '.' and '.' in tokens[i]):
            tokens[i] = tokens[i].replace('.','')

        while (tokens[i] != '\'' and '\'' in tokens[i]):
            tokens[i] = tokens[i].replace('\'','')

        while (tokens[i] != '~' and '~' in tokens[i]):
            tokens[i] = tokens[i].replace('~','')

        while('-' in tokens[i] or '/' in tokens[i]):
            if tokens[i] == '/' or tokens[i] == '-':
                tokens.pop(i)
                i -= 1
            if '/' in tokens[i]:
                split = tokens[i].split('/')
                tokens[i] = split[0]
                i += 1
                tokens.insert(i, split[1])
            if '-' in tokens[i]:
                split = tokens[i].split('-')                
                tokens[i] = split[0]
                i += 1
                tokens.insert(i, split[1])
            if tokens[i-1] == '/' or tokens[i-1] == '-':
                tokens.pop(i-1)
                i -= 1
            if '/' in tokens[i-1]:
                split = tokens[i-1].split('/')
                tokens[i-1] = split[0]
                i += 1
                tokens.insert(i-1, split[1])
            if '-' in tokens[i-1]:
                split = tokens[i-1].split('-')                
                tokens[i-1] = split[0]
                i += 1
                tokens.insert(i-1, split[1])
        
        if tokens[i].endswith('er'):
            tokens[i] = tokens[i][:-2]
            i += 1
            tokens.insert(i, 'er')
        
        if tokens[i].endswith('est'):
            tokens[i] = tokens[i][:-3]
            i += 1
            tokens.insert(i, 'est')
        
        if tokens[i].endswith('ish'):
            tokens[i] = tokens[i][:-3]
            i += 1
            tokens.insert(i, 'ish')
        
        if tokens[i-1].endswith('er'):
            tokens[i-1] = tokens[i-1][:-2]
            i += 1
            tokens.insert(i-1, 'er')
        
        if tokens[i-1].endswith('est'):
            tokens[i-1] = tokens[i-1][:-3]
            i += 1
            tokens.insert(i-1, 'est')
        
        if tokens[i-1].endswith('ish'):
            tokens[i-1] = tokens[i-1][:-3]
            i += 1
            tokens.insert(i-1, 'ish')
        
        i += 1
    
    replace = {
        'redd':'red', 
        'gren': 'green',
        'whit':'white',
        'biege':'beige',
        'purp':'purple',
        'olve':'olive',
        'ca':'can',
        'blu':'blue',
        'orang':'orange',
        'gray':'grey',
    }
    
    for i in range(len(tokens)):
        if tokens[i] in replace.keys():
            tokens[i] = replace[tokens[i]]
    
    while '' in tokens:
        tokens.remove('')
    
    return tokens