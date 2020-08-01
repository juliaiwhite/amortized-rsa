import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
from torchsummary import summary
import data

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

class FeatureMLP(nn.Module):
    def __init__(self, input_size=16, output_size=16):
        super(FeatureMLP, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        return self.trunk(x)


def to_onehot(y, n=3):
    y_onehot = torch.zeros(y.shape[0], n).to(y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


class Speaker(nn.Module):
    def __init__(self, feat_model, embedding_module, hidden_size=100):
        super(Speaker, self).__init__()
        self.embedding = embedding_module
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.hidden_size = hidden_size

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        # n_obj of feature size + 1/0 indicating target index
        self.init_h = nn.Linear(3 * (self.feat_size + 1), self.hidden_size)

    def embed_features(self, feats, targets):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.view(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)

        # Add targets
        targets_onehot = to_onehot(targets)

        feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)

        ft_concat = feats_and_targets.view(batch_size, -1)
        
        return ft_concat

    def forward(self, feats, targets, greedy=False, activation='gumbel', tau = 1, length_penalty=False, max_len=40):
        """Sample from image features"""
        batch_size = feats.size(0)

        feats_emb = self.embed_features(feats, targets)

        # initialize hidden states using image features
        states = self.init_h(feats_emb)
        states = states.unsqueeze(0)

        # This contains are series of sampled onehot vectors
        lang = []
        if length_penalty:
            eos_prob = []
            
        if activation == 'multinomial':
            lang_prob = []
        else:
            lang_prob = None
        
        # And vector lengths
        lang_length = torch.ones(batch_size, dtype=torch.int64).to(feats.device)
        done_sampling = [False for _ in range(batch_size)]

        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size).to(feats.device)
        inputs_onehot[:, data.SOS_IDX] = 1.0

        # (batch_size, len, n_vocab)
        inputs_onehot = inputs_onehot.unsqueeze(1)

        # Add SOS to lang
        lang.append(inputs_onehot)

        # (B,L,D) to (L,B,D)
        inputs_onehot = inputs_onehot.transpose(0, 1)

        # compute embeddings
        # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        inputs = inputs_onehot @ self.embedding.weight

        for i in range(max_len - 2):  # Have room for SOS, EOS if never sampled
            # FIXME: This is inefficient since I do sampling even if we've
            # finished generating language.
            if all(done_sampling):
                break
            self.gru.flatten_parameters()
            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)                # outputs: (B,H)
            outputs = self.outputs2vocab(outputs)       # outputs: (B,V)
            
            if greedy:
                predicted = outputs.max(1)[1]
                predicted = predicted.unsqueeze(1)
            else:
                #  outputs = F.softmax(outputs, dim=1)
                #  predicted = torch.multinomial(outputs, 1)
                # TODO: Need to let language model accept one-hot vectors.
                if activation=='gumbel'or activation==None:
                    predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=True)
                elif activation=='softmax':
                    predicted_onehot = F.softmax(outputs/tau)
                elif activation=='softmax_noise':
                    predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=False)
                elif activation == 'multinomial':
                    # Normal non-differentiable sampling from the RNN, trained with REINFORCE
                    TEMP = 5.0
                    idx_prob = F.log_softmax(outputs / TEMP, dim=1)
                    predicted = torch.multinomial(idx_prob.exp(), 1)
                    predicted_onehot = to_onehot(predicted, n=self.vocab_size)
                    predicted_logprob = torch.gather(idx_prob, 1, predicted)
                    lang_prob.append(predicted_logprob)
                else:
                    raise NotImplementedError(activation)
                    
                # Add to lang
                lang.append(predicted_onehot.unsqueeze(1))
                if length_penalty:
                    idx_prob = F.log_softmax(outputs, dim = 1)
                    eos_prob.append(idx_prob[:,data.EOS_IDX])

            predicted_npy = predicted_onehot.argmax(1).cpu().numpy()
            
            # Update language lengths
            for j, pred in enumerate(predicted_npy):
                if not done_sampling[j]:
                    lang_length[j] += 1
                if pred == data.EOS_IDX and activation in {'gumbel', 'multinomial'}:
                    done_sampling[j] = True

            # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
            inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

        # If multinomial, we need to run inputs once more to get the logprob of
        # EOS (in case we've sampled that far)
        if activation == 'multinomial':
            self.gru.flatten_parameters()
            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)                # outputs: (B,H)
            outputs = self.outputs2vocab(outputs)       # outputs: (B,V)
            idx_prob = F.log_softmax(outputs, dim=1)
            lang_prob.append(idx_prob[:, data.EOS_IDX].unsqueeze(1))
            
        # Add EOS if we've never sampled it
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(feats.device)
        eos_onehot[:, 0, data.EOS_IDX] = 1.0
        lang.append(eos_onehot)
        
        # Cut off the rest of the sentences
        for i, _ in enumerate(predicted_npy):
            if not done_sampling[i]:
                lang_length[i] += 1
            done_sampling[i] = True

        # Cat language tensors
        lang_tensor = torch.cat(lang, 1)
        
        for i in range(lang_tensor.shape[0]):
            lang_tensor[i, lang_length[i]:] = 0

        # Trim max length
        max_lang_len = lang_length.max()
        lang_tensor = lang_tensor[:, :max_lang_len, :]
        
        if activation == 'multinomial':
            lang_prob_tensor = torch.cat(lang_prob, 1)
            for i in range(lang_prob_tensor.shape[0]):
                lang_prob_tensor[i, lang_length[i]:] = 0
            lang_prob_tensor = lang_prob_tensor[:, :max_lang_len]
            lang_prob = lang_prob_tensor.sum(1)
        
        if length_penalty:
            # eos prob -> eos loss
            eos_prob = torch.stack(eos_prob, dim = 1)
            for i in range(eos_prob.shape[0]):
                r_len = torch.arange(1,eos_prob.shape[1]+1,dtype=torch.float32)
                eos_prob[i] = eos_prob[i]*r_len.to(eos_prob.device)
                eos_prob[i, lang_length[i]:] = 0
            eos_loss = -eos_prob
            eos_loss = eos_loss.sum(1)/lang_length.float()
            eos_loss = eos_loss.mean()
        else:
            eos_loss = 0
            
        # Sum up log probabilities of samples
        return lang_tensor, lang_length, eos_loss, lang_prob

    def to_text(self, lang_onehot):
        texts = []
        lang = lang_onehot.argmax(2)
        for sample in lang.cpu().numpy():
            text = []
            for item in sample:
                text.append(data.ITOS[item])
                if item == data.EOS_IDX:
                    break
            texts.append(' '.join(text))
        return np.array(texts, dtype=np.unicode_)


class LiteralSpeaker(nn.Module):
    def __init__(self, feat_model, embedding_module, hidden_size=100):
        super(LiteralSpeaker, self).__init__()
        self.contextual = True
        
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.hidden_size = hidden_size
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        if self.contextual:
            self.init_h = nn.Linear(3 * (self.feat_size + 1), self.hidden_size)
        else:
            self.init_h = nn.Linear(self.feat_size, self.hidden_size)
        
    def forward(self, feats, seq, length, y):
        try: self.contextual
        except:
            self.contextual = False
        batch_size = seq.shape[0]
        if self.contextual:
            n_obj = feats.shape[1]
            rest = feats.shape[2:]
            feats = feats.view(batch_size * n_obj, *rest)
            feats_emb_flat = self.feat_model(feats)
            feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
            # Add targets
            targets_onehot = to_onehot(y)
            feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)
            feats_emb = feats_and_targets.view(batch_size, -1)
        else:
            feats = torch.from_numpy(np.array([np.array(feat[y[idx],:,:,:].cpu()) for idx, feat in enumerate(feats)])).cuda()
            feats_emb = self.feat_model(feats.cuda())
            
        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1).to(feats.device)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight
            
        #feats_emb = self.feat_model(feats.squeeze().to(feats.device))
        feats_emb = self.init_h(feats_emb)
        feats_emb = feats_emb.unsqueeze(0)
        
        # shape = (seq_len, batch, hidden_dim)
        self.gru.flatten_parameters()
        output, _ = self.gru(embed_seq, feats_emb)

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)

        max_length = output.size(1)
        output_2d = output.reshape(batch_size * max_length, -1)
        outputs_2d = self.outputs2vocab(output_2d)
        lang_tensor = outputs_2d.reshape(batch_size, max_length, self.vocab_size)
        return lang_tensor
    
    def sample(self, feats, y, greedy=False):
        """Generate from image features using greedy search."""
        try: self.contextual
        except:
            self.contextual = False
        with torch.no_grad():
            batch_size = feats.shape[0]
            if self.contextual:
                n_obj = feats.shape[1]
                rest = feats.shape[2:]
                feats = feats.view(batch_size * n_obj, *rest)
                feats_emb_flat = self.feat_model(feats)
                feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
                # Add targets
                targets_onehot = to_onehot(y)
                feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)
                feats_emb = feats_and_targets.view(batch_size, -1)
            else:
                feats = torch.from_numpy(np.array([np.array(feat[y[idx],:,:,:].cpu()) for idx, feat in enumerate(feats)])).cuda()
                feats_emb = self.feat_model(feats)

            # initialize hidden states using image features
            feats_emb = self.init_h(feats_emb)
            states = feats_emb.unsqueeze(0)

            # first input is SOS token
            max_len = 40
            inputs = np.array([SOS_IDX for _ in range(batch_size)])
            inputs = torch.from_numpy(inputs)
            inputs = inputs.unsqueeze(1)
            inputs = inputs.to(feats.device)
            inputs = F.one_hot(inputs, num_classes=self.vocab_size).float()

            # save SOS as first generated token
            inputs_npy = inputs.squeeze(1).cpu().numpy()
            sampled = np.array([[w] for w in inputs_npy])
            sampled = np.transpose(sampled, (1, 0, 2))

            # (B,L,D) to (L,B,D)
            inputs = inputs.transpose(0,1).to(feats.device)

            # compute embeddings
            inputs = inputs @ self.embedding.weight

            for i in range(max_len-1):
                self.gru.flatten_parameters()
                outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(0)                # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1].cpu()
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs.cpu(), 1)
                    
                predicted = predicted.transpose(0, 1)        # inputs: (L=1,B)
                predicted = F.one_hot(predicted, num_classes=self.vocab_size).float()
                inputs = predicted.to(feats.device) @ self.embedding.weight             # inputs: (L=1,B,E)
                
                sampled = np.concatenate((sampled,predicted),axis = 0)
            
            sampled = torch.tensor(sampled).permute(1,0,2)
            
            sampled_id = sampled.reshape(sampled.shape[0]*sampled.shape[1],-1).argmax(1).reshape(sampled.shape[0],sampled.shape[1])
            sampled_lengths = torch.tensor([np.count_nonzero(t) for t in sampled_id.cpu()], dtype=np.int)
        return sampled, sampled_lengths
    
class LanguageModel(nn.Module):
    def __init__(self, embedding_module, hidden_size=100):
        super(LanguageModel, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.vocab_size = embedding_module.num_embeddings
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, seq, length):
        batch_size = seq.shape[0]
        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq.cuda() @ self.embedding.weight
        
        # shape = (seq_len, batch, hidden_dim)
        feats_emb = torch.zeros(1, batch_size, self.hidden_size).to(embed_seq.device)
        self.gru.flatten_parameters()
        output, _ = self.gru(embed_seq, feats_emb)

        # reorder from (L,B,D) to (B,L,D)
        output = output.transpose(0, 1)

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            output = output[reversed_idx]

        max_length = output.size(1)
        output_2d = output.view(batch_size * max_length, -1)
        outputs_2d = self.outputs2vocab(output_2d)
        lang_tensor = outputs_2d.view(batch_size, max_length, self.vocab_size)
        return lang_tensor
    
    def probability(self, seq, length):
        with torch.no_grad():
            max_len = 40
            batch_size = seq.shape[0]
            seq = F.pad(seq,(0,0,0,(max_len-seq.shape[1]))).float()
            # reorder from (B,L,D) to (L,B,D)
            seq = seq.transpose(0, 1)
            # embed your sequences
            embed_seq = seq.cuda() @ self.embedding.weight

            # shape = (seq_len, batch, hidden_dim)
            feats_emb = torch.zeros(1, batch_size, self.hidden_size).to(embed_seq.device)
            
            inputs = embed_seq
            states = feats_emb
            prob = torch.zeros(batch_size)
            
            self.gru.flatten_parameters()
            outputs, _ = self.gru(inputs, states)
            outputs = outputs.squeeze(0)
            outputs = self.outputs2vocab(outputs)
            
            idx_prob = F.log_softmax(outputs,dim=2).cpu().numpy()
            for word_idx in range(1,seq.shape[0]):
                for utterance_idx, word in enumerate(seq[word_idx].argmax(1)):
                    if word_idx < length[utterance_idx]:
                        prob[utterance_idx] = prob[utterance_idx]+idx_prob[word_idx-1,utterance_idx,word]/length[utterance_idx]
        return prob
    
class RNNEncoder(nn.Module):
    """
    RNN Encoder - takes in onehot representations of tokens, rather than numeric
    """
    def __init__(self, embedding_module, hidden_size=100):
        super(RNNEncoder, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq.cuda() @ self.embedding.weight

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        self.gru.flatten_parameters()
        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden

class Listener(nn.Module):
    def __init__(self, feat_model, embedding_module):
    
        super(Listener, self).__init__()
        self.embedding = embedding_module
        self.lang_model = RNNEncoder(self.embedding)
        self.feat_model = feat_model
        self.feat_size = feat_model.final_feat_dim
        self.vocab_size = embedding_module.num_embeddings
        self.bilinear = nn.Linear(self.lang_model.hidden_size, self.feat_size, bias=False)

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)
        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        return feats_emb

    def forward(self, feats, lang, lang_length, average=False):
        max_len=40
        
        if average:
            weights = torch.mul(1/((1-lang[:,:,EOS_IDX]).sum(1).unsqueeze(1).repeat(1,40)),(1-lang[:,:,EOS_IDX])).unsqueeze(1).repeat(1,3,1)
            scores = 0
            for i in range(0,max_len-1):
                # Embed features
                feats_emb = self.embed_features(feats)

                # Embed language
                lang_emb = self.lang_model(lang, lang_length-i)

                # Bilinear term: lang embedding space -> feature embedding space
                lang_bilinear = self.bilinear(lang_emb)

                # Compute dot products
                #scores = (scores+torch.einsum('ijh,ih->ij', (feats_emb, lang_bilinear)))
                scores = scores+torch.mul(weights[:,:,i],torch.einsum('ijh,ih->ij', (feats_emb, lang_bilinear)))
                
        else:
            # Embed features
            feats_emb = self.embed_features(feats)
            
            # Embed language
            lang_emb = self.lang_model(lang, lang_length)

            # Bilinear term: lang embedding space -> feature embedding space
            lang_bilinear = self.bilinear(lang_emb)

            # Compute dot products
            scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_bilinear)))
        return scores