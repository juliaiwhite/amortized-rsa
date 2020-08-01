"""
"""

import numpy as np
from sklearn.model_selection import train_test_split
import random
import string
import shapeworld


PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<UNK>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def init_vocab(langs):
    i2w = {
        PAD_IDX: PAD_TOKEN,
        SOS_IDX: SOS_TOKEN,
        EOS_IDX: EOS_TOKEN,
        UNK_IDX: UNK_TOKEN,
    }
    w2i = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for lang in langs:
        for tok in lang:
            if tok not in w2i:
                i = len(w2i)
                w2i[tok] = i
                i2w[i] = tok
    return {'w2i': w2i, 'i2w': i2w}


def train_val_test_split(data,
                         val_size=0.1,
                         test_size=0.1,
                         random_state=None):
    """
    Split data into train, validation, and test splits
    Parameters
    ----------
    data : ``np.Array``
        Data of shape (n_data, 2), first column is ``x``, second column is ``y``
    val_size : ``float``, optional (default: 0.1)
        % to reserve for validation
    test_size : ``float``, optional (default: 0.1)
        % to reserve for test
    random_state : ``np.random.RandomState``, optional (default: None)
        If specified, random state for reproducibility
    """
    idx = np.arange(data['imgs'].shape[0])
    idx_train, idx_valtest = train_test_split(idx,
                                              test_size=val_size + test_size,
                                              random_state=random_state,
                                              shuffle=True)
    idx_val, idx_test = train_test_split(idx_valtest,
                                         test_size=test_size /
                                         (val_size + test_size),
                                         random_state=random_state,
                                         shuffle=True)
    splits = []
    for idx_split in (idx_train, idx_val, idx_test):
        splits.append({
            'imgs': data['imgs'][idx_split],
            'labels': data['labels'][idx_split],
            'langs': data['langs'][idx_split],
        })
    return splits

def load_raw_data(data_file):
    data = np.load(data_file)
    # Preprocessing/tokenization
    try:
        return {
            'imgs': data['imgs'].transpose(0, 1, 4, 2, 3),
            'labels': data['labels'],
            'langs': np.array([t.lower().split() for t in data['langs']])
        }
    except:
        return {
            'imgs': data['imgs'],
            'labels': data['labels'],
            'langs': data['langs']
        }

class ShapeWorld:
    def __init__(self, data, vocab):
        self.imgs = data['imgs']
        self.labels = data['labels']
        # Get vocab
        self.w2i = vocab['w2i']
        self.i2w = vocab['i2w']
        if len(vocab['w2i']) > 100:
            self.lang_raw = data['langs']
            self.lang_idx = data['langs']
        else:
            self.lang_raw = data['langs']
            self.lang_idx, self.lang_len = self.to_idx(self.lang_raw)

    def __len__(self):
        return len(self.lang_raw)

    def __getitem__(self, i):
        # Reference game format.
        img = self.imgs[i]
        label = self.labels[i]
        lang = self.lang_idx[i]
        return (img, label, lang)

    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t) for t in langs], dtype=np.int) + 2
        lang_idx = np.full((len(self), max(lang_len)), self.w2i[PAD_TOKEN], dtype=np.int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks, start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len