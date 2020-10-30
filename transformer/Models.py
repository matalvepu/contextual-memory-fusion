''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F


__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    """If the seq is of shape (num_batch,seq_len,n_features), returns a tensor of shape (num_batch,seq_len)
    where a entry is 1 if it is not a padding entry, 0 otherwise
    """
    
    
    numpy_seq_k = seq.detach().cpu().numpy()
    #print(numpy_seq_k[1,:10,:])
    padding_rows_cols = np.where(~numpy_seq_k.any(axis=2))
    padding_mask = np.ones((numpy_seq_k.shape[:2]))
   
    padding_mask[padding_rows_cols[0],padding_rows_cols[1]]=0
    return torch.from_numpy(padding_mask).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    """If the seq is of shape (num_batch,seq_len,n_features), returns a tensor of shape (num_batch,seq_len,seq_len)
    where a entry is 0 if it is not a padding entry, 1 otherwise. Note that a vector of length seq_len is repeated 
    seq_len times.
    """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)#20
    #calling detach will have no impact on the original seq_q,
    numpy_seq_k = seq_k.detach().cpu().numpy()
    #print("checking state of seq_k after detach() op",seq_k.requires_grad)
    #print(numpy_seq_k[1,:10,:])
    padding_rows_cols = np.where(~numpy_seq_k.any(axis=2))
    padding_mask = np.zeros((numpy_seq_k.shape[:2]))
    #MY best guess is that There is zero in non-padding entries and one is padding entries.
    padding_mask[padding_rows_cols[0],padding_rows_cols[1]]=1
    padding_mask = torch.from_numpy(padding_mask).unsqueeze(1).expand(-1, len_q, -1)
    #print(padding_mask,padding_mask.size())
    #np.where(~a.any(axis=1))[0]
    #padding_mask = seq_k.eq(Constants.PAD)
    #padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_features, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Encoder,self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Linear(
            n_src_features, d_model)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self,X,X_pos, device, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=X, seq_q=X)
        non_pad_mask = get_non_pad_mask(seq = X)
        
        slf_attn_mask = slf_attn_mask.type(torch.ByteTensor).to(device)
        non_pad_mask = non_pad_mask.type(torch.FloatTensor).to(device)
        #print("src_seq:",src_seq.size())
        #print("slf_attn_mask:",slf_attn_mask.size())
        #print(" non_pad_mask:",non_pad_mask.size())

        # -- Forward
        enc_output = self.src_word_emb(X) + self.position_enc(X_pos)
        #print("enc_output:",enc_output.size())

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            #print("enc_slf_attn:",enc_slf_attn.size())
            
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class PostEncoder(nn.Module):
    '''
    This module will merge all the modified "feature vectors" derived from the encoder
    '''
    
    def __init__(self,_config):
        super(PostEncoder,self).__init__()
        self.device = _config["device"]
        self.post_merger_config = _config["post_merger"]
      
        
        self.fc1 = nn.Linear(in_features=self.post_merger_config["lstm_hidden"],
                             out_features=self.post_merger_config["fc1_output"])
        
        self.fc1_dropout = nn.Dropout(self.post_merger_config["dropout"])
        
        self.fc2 = nn.Linear(in_features=self.post_merger_config["fc1_output"],
                             out_features=self.post_merger_config["fc2_output"])
    def forward(self,enc_output):
        #print("enc_output_size:",enc_output.size())
        
        batch_size = enc_output.size()[0]
        hidden_size = self.post_merger_config["lstm_hidden"]
        
        h_l = torch.zeros(batch_size, hidden_size).unsqueeze(0).to(self.device)
        c_l = torch.zeros(batch_size, hidden_size).unsqueeze(0).to(self.device)
        
        _,(h_last,c_last) = self.merger_lstm(enc_output,(h_l,c_l))
        #print("h_last_size:",h_last.size())
        ret_val =  self.fc2(self.fc1_dropout(F.relu(self.fc1(h_last))))
        #print(ret_val.size())
        return ret_val
        
        
class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_features, len_max_seq,_config,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):
        #We are choosing d_model and setting d_word_vec as the same value
        d_word_vec = d_model

        super(Transformer,self).__init__()
        self.device = _config["device"]

        self.encoder = Encoder(
            n_src_features=n_src_features, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        #TODO:Needs to uncomment it
        #self.post_merger = PostEncoder(_config)
        in_size = d_model*_config["num_context_sequence"]
        out_size = _config["mfn_configs"][0]["memsize"]
        self.fc_to_mfn_mem = nn.Linear(in_size,out_size)
        
        
        
        
        

       
       

    def forward(self,X,X_pos,Y):

        #tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(X,X_pos,self.device)
        #TODO:Need to uncomment it
        #predictions = self.post_merger(enc_output)
        #print("output from transformer:",enc_output.shape)
        
        reshaped_enc_out = torch.reshape(enc_output,(enc_output.shape[0],-1))
        mfn_mem_lstm_input = self.fc_to_mfn_mem(reshaped_enc_out)
        #print("mfn mem input shape:",mfn_mem_lstm_input.size())      

        return mfn_mem_lstm_input
