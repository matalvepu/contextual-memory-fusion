import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import h5py
import time
import data_loader as loader
from collections import defaultdict, OrderedDict
import argparse
import pickle as pickle
import time
import json, os, ast, h5py


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

my_logger=None


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class MFN(nn.Module):
    def __init__(self,_config):
        
        super(MFN, self).__init__()
        self.config=_config
        self.device = _config["device"]
        config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig = \
            _config["mfn_configs"]
        [self.d_l_orig,self.d_a,self.d_v,self.d_l_embedded] = config["input_dims"]
        [self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
        total_h_dim = self.dh_l+self.dh_a+self.dh_v
        self.mem_dim = config["memsize"]
        window_dim = config["windowsize"]
        output_dim = 1
        attInShape = total_h_dim*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_h_dim+self.mem_dim
        h_att1 = NN1Config["shapes"]
        h_att2 = NN2Config["shapes"]
        h_gamma1 = gamma1Config["shapes"]
        h_gamma2 = gamma2Config["shapes"]
        h_out = outConfig["shapes"]
        att1_dropout = NN1Config["drop"]
        att2_dropout = NN2Config["drop"]
        gamma1_dropout = gamma1Config["drop"]
        gamma2_dropout = gamma2Config["drop"]
        out_dropout = outConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l_embedded, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)
        
        self.word_embedding = Word_Embedding(_config)
        
        
        
        
    
    def forward(self,x,c_l_prior,c_a_prior,c_v_prior,c_mem_prior):
        
        x_l = self.word_embedding(x[:,:,:self.d_l_orig])
        x_a = x[:,:,self.d_l_orig:self.d_l_orig+self.d_a]
        x_v = x[:,:,self.d_l_orig+self.d_a:]
        
        #if we do not need the entire punchline, we will zero out everything
        if(self.config["use_punchline"]==False):
            x_l = torch.zeros_like(x_l,requires_grad=True)
            x_a = torch.zeros_like(x_a,requires_grad=True)
            x_v = torch.zeros_like(x_v,requires_grad=True)


        
        #If we do not need to use punchline text, we can zero it out
        if(self.config["use_punchline_text"]==False):
            x_l = torch.zeros_like(x_l,requires_grad=True)
        
        
        #If we do not need to use punchline audio, we can zero it out
        if(self.config["use_punchline_audio"]==False):
            x_a = torch.zeros_like(x_a,requires_grad=True)
            #my_logger.debug("The zeroed audio:",x_l)
        
         #If we do not need to use punchline video, we can zero it out
        if(self.config["use_punchline_video"]==False):
            x_v = torch.zeros_like(x_v,requires_grad=True)
            #my_logger.debug("The zeroed video:",x_v)
        #Here, we will check selective audio/visual removing
        if(self.config["selectively_omitted_index"] !=-1):
            feat_index = self.config["selectively_omitted_index"]
            feat_entry = self.config["selective_audio_visual_feature_omission"][feat_index]
            print("removing:",feat_entry["name"])
            if(feat_entry["modality"]=="audio"):
                x_a[:,:,feat_entry["indices"]]=0.0
            elif(feat_entry["modality"]=="video"):
                x_v[:,:,feat_entry["indices"]]=0.0

                
            
        
        

        
        
        # x is t x n x d
        n = x.shape[1]
        t = x.shape[0]
        self.h_l = torch.zeros(n, self.dh_l).to(self.device)
        self.h_a = torch.zeros(n, self.dh_a).to(self.device)
        self.h_v = torch.zeros(n, self.dh_v).to(self.device)
        #My best guess is that we need to initialize c with the prior, not h. BUt I can be wrong.
        #Talk to Amir about it
# =============================================================================
#         self.c_l = torch.zeros(n, self.dh_l).to(self.device)
#         self.c_a = torch.zeros(n, self.dh_a).to(self.device)
#         self.c_v = torch.zeros(n, self.dh_v).to(self.device)
#         self.mem = torch.zeros(n, self.mem_dim).to(self.device)
# =============================================================================
        self.c_l = c_l_prior.to(self.device)
        self.c_a = c_a_prior.to(self.device)
        self.c_v = c_v_prior.to(self.device)
        self.mem = c_mem_prior.to(self.device)
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(t):
            # prev time step
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v
            # curr time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = torch.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            cHat = torch.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        
        #This new line runs the sigmoid and gives 0/1 output. Our losee function takes care of that
        #prediction = torch.sigmoid(output)>=0.5
        return output

class Word_Embedding(nn.Module):
    def __init__(self,_config):
        
        super(Word_Embedding, self).__init__()  
        
        word_emb_list_file=os.path.join(_config["dataset_location"],"humor_word_embedding_list.pkl")
        humor_word_emb_list=load_pickle(word_emb_list_file)
        
        vocab=torch.LongTensor(humor_word_emb_list)
        self.embed = nn.Embedding(len(vocab),len(vocab[0]))
        self.embed.weight.data.copy_(vocab)
        self.embed.weight.requires_grad = False
        
    def forward(self,X_index):
        #print("We got as X index",X_index.shape)
        #it returns a batc,seq_len,1,300 dim vector bt our calculation expects a batch,seq_len,300d vector
        return self.embed(X_index.long()).squeeze(-2)


 
class Unimodal_Context(nn.Module):
    def __init__(self,_config):
        super(Unimodal_Context, self).__init__()

        relevant_config = _config["unimodal_context"]
        #print("Unimodal configs:",relevant_config)
        #TODO: Must change it id text is sent as embedding. ANother way is to make the change in config file directly
        [self.h_text,self.h_audio,self.h_video] = relevant_config["hidden_sizes"]
        self.text_LSTM = nn.LSTM(input_size = relevant_config["text_lstm_input"],
                    hidden_size = self.h_text,
                    batch_first=True)
        self.audio_LSTM = nn.LSTM(input_size = relevant_config["audio_lstm_input"],
                    hidden_size = self.h_audio,
                    batch_first=True)
        self.video_LSTM  = nn.LSTM(input_size = relevant_config["video_lstm_input"],
                    hidden_size = self.h_video,
                    batch_first=True)
        self.device = _config["device"]
        #self.hidden_size = relevant_config["hidden_size"]
        self.input_dims = _config["input_dims"]
        self.config=_config
        
        self.word_embedding = Word_Embedding(_config)
        
    def forward(self,X_context):
        old_batch_size,context_size,seq_len,num_feats = X_context.size()
        
        # #As LSTM accepts only (batch,seq_len,feats), we are reshaping the tensor.However,
        # #it should not have any problem. There may be some issues during backprop, but lets see what happens
        
        X_context = torch.reshape(X_context,(old_batch_size*context_size,seq_len,num_feats)).to(self.device)
        
        new_batch_size = old_batch_size*context_size

        #my_logger.debug("\nX_context:",X_context.size())
        #The X_Context entries do not have a 300 length embedding, only an index is present.
        #we will substiture the index with the 300-D vector
        text_context = self.word_embedding(X_context[:,:,:self.input_dims[0]])
        audio_context = X_context[:,:,self.input_dims[0]:self.input_dims[0]+self.input_dims[1]]
        video_context = X_context[:,:,self.input_dims[0]+self.input_dims[1]:]
        
        #If we do not need to use context text, we can zero it out
        if(self.config["use_context_text"]==False):
             text_context= torch.zeros_like(text_context ,requires_grad=True)
        
        
        #If we do not need to use context audio, we can zero it out
        if(self.config["use_context_audio"]==False):
            audio_context = torch.zeros_like(audio_context,requires_grad=True)
            #my_logger.debug("The zeroed audio:",x_l)
        
        #If we do not need to use context video, we can zero it out
        if(self.config["use_punchline_video"]==False):
            video_context = torch.zeros_like(video_context,requires_grad=True)
            #my_logger.debug("The zeroed video:",x_v)
            
        #we can remove features selectively too
        if(self.config["selectively_omitted_index"] !=-1):
            feat_index = self.config["selectively_omitted_index"]
            feat_entry = self.config["selective_audio_visual_feature_omission"][feat_index]
            #print("removing:",feat_entry["name"])
            if(feat_entry["modality"]=="audio"):
                audio_context[:,:,feat_entry["indices"]]=0.0
            elif(feat_entry["modality"]=="video"):
                video_context[:,:,feat_entry["indices"]]=0.0
            
        
        #print("Context shapes:\n","t:",text_context.shape,"a:",audio_context.shape,"v:",video_context.shape)

        
       
        #The text lstm
        ht_l = torch.zeros(new_batch_size, self.h_text).unsqueeze(0).to(self.device)
        ct_l = torch.zeros(new_batch_size, self.h_text).unsqueeze(0).to(self.device)
        _,(ht_last,ct_last) = self.text_LSTM(text_context,(ht_l,ct_l))
        #my_logger.debug("ht_last:",ht_last.shape)
        
        
        ha_l = torch.zeros(new_batch_size, self.h_audio).unsqueeze(0).to(self.device)
        ca_l = torch.zeros(new_batch_size, self.h_audio).unsqueeze(0).to(self.device)
        _,(ha_last,ca_last) = self.audio_LSTM(audio_context,(ha_l,ca_l))
        #my_logger.debug("ha_last:",ha_last.shape)
        
        hv_l = torch.zeros(new_batch_size, self.h_video).unsqueeze(0).to(self.device)
        cv_l = torch.zeros(new_batch_size, self.h_video).unsqueeze(0).to(self.device)
        _,(hv_last,cv_last) = self.video_LSTM(video_context,(hv_l,cv_l))
        #my_logger.debug("ha last:",hv_last.shape)
        
        text_lstm_result = torch.reshape(ht_last,(old_batch_size,context_size,-1))
        audio_lstm_result = torch.reshape(ha_last,(old_batch_size,context_size,-1))
        video_lstm_result = torch.reshape(hv_last,(old_batch_size,context_size,-1))
        #my_logger.debug("final result from unimodal:",text_lstm_result.shape,audio_lstm_result.shape,video_lstm_result.shape)

        
        return text_lstm_result,audio_lstm_result,video_lstm_result
        


class Multimodal_Context(nn.Module):
    def __init__(self,_config):
        super(Multimodal_Context, self).__init__()
        #print("Config in multimodal context:",_config["multimodal_context_configs"])
        self.config = _config
        (in_text,in_audio,in_video) =  [ _config["num_context_sequence"]*e for e in _config["unimodal_context"]["hidden_sizes"]]
        
        #mfn config contains a list of configs and the first one of them is the config, which
        #contains a dictionary called h_dims which has the [ht,ha,hv].
        (out_text,out_audio,out_video) = _config["mfn_configs"][0]["h_dims"]
        
        #The first one is hl
        self.fc_uni_text_to_mfn_text_input = nn.Linear(in_text,out_text)
        self.text_in_drop = nn.Dropout(_config["multimodal_context_configs"]["text_in_drop"])
        
        #The second one is ha
        self.fc_uni_audio_to_mfn_audio_input = nn.Linear(in_audio,out_audio)
        self.audio_in_drop = nn.Dropout(_config["multimodal_context_configs"]["audio_in_drop"])

        
        #The third one is hv
        self.fc_uni_video_to_mfn_video_input = nn.Linear(in_video,out_video)
        self.video_in_drop = nn.Dropout(_config["multimodal_context_configs"]["video_in_drop"])

        
        #This one will output the initialization of the mfn meory
        encoder_config =self.config["multimodal_context_configs"]
        self.self_attention_module = Transformer(
        
        n_src_features = encoder_config["n_source_features"],
        len_max_seq = encoder_config["max_token_seq_len"],
        _config = self.config,
        tgt_emb_prj_weight_sharing=encoder_config["proj_share_weight"],
        emb_src_tgt_weight_sharing=encoder_config["embs_share_weight"],
        d_k=encoder_config["d_k"],
        d_v=encoder_config["d_v"],
        d_model=encoder_config["d_model"],
        d_word_vec=encoder_config["d_word_vec"],
        d_inner=encoder_config["d_inner_hid"],
        n_layers=encoder_config["n_layers"],
        n_head=encoder_config["n_head"],
        dropout=encoder_config["dropout"]
        ).to(self.config["device"])
        
        self.mem_in_drop = nn.Dropout(_config["multimodal_context_configs"]["mem_in_drop"])


    
    def forward(self,text_uni,audio_uni,video_uni,X_pos_Context,Y):
        #So, we are getting three tensor corresponding to three modalities, each of shape:torch.Size([10, 5, 64])
        
        #We will initialize the text lstm of mfn solely from the result of text_uni.
        
        #Text_uni has shape [10,5,64], we will convert need to convert it to [batch_size,hidden_size].\
        #So, first, we can just convert it to [10,5*64] here 10 is the batch size.
        #The same is done with audio and video uni
        reshaped_text_uni = text_uni.reshape((text_uni.shape[0],-1))
        #my_logger.debug("reshaped text:",reshaped_text_uni.shape)
        reshaped_audio_uni = audio_uni.reshape((audio_uni.shape[0],-1))
        #my_logger.debug("reshaped audio:",reshaped_audio_uni.shape)
        reshaped_video_uni = video_uni.reshape((video_uni.shape[0],-1))
        #my_logger.debug("reshaped video:",reshaped_video_uni.shape)
        
        #Then, we will have three linear trans. So, all three reshaped tensors begin with 
        #shape (batch_size,config.num_context_sequence*config.unimodal_context.hidden_size) 
        #And we need to convert them to (batch_size,mfn_configs.config.[hl or ht or hv])
        #ht means hidden text
        #TODO: May use a dropout layer later
        mfn_hl_input = self.text_in_drop(self.fc_uni_text_to_mfn_text_input(reshaped_text_uni))
        #ha means hidden audio
        mfn_ha_input = self.audio_in_drop(self.fc_uni_audio_to_mfn_audio_input(reshaped_audio_uni))
        #hv means hidden video
        mfn_hv_input = self.video_in_drop(self.fc_uni_video_to_mfn_video_input(reshaped_video_uni))
        #These three will be used to initialize the three unimodal lstms of mfn
        #my_logger.debug("mfn text lstm hidden init:",mfn_ht_input.shape)
        #my_logger.debug("mfn audio lstm hidden init:",mfn_ha_input.shape)
        #my_logger.debug("mfn video lstm hidden init:",mfn_hv_input.shape)
        
        
        #Now, we will do self attention to convert all three original text_uni,audio_uni and video_uni 
        #to feed into transformer. They are of shape (10,5,64), (10,5,8) and (10,5,16). SO, we need to first concat them 
        #to convert them to shape (20,5,64+8+16=88). So, we will concat them by axis=2
        all_three_orig_concat = torch.cat([text_uni,audio_uni,video_uni],dim=2)
        my_logger.debug("all mods concatenated:",all_three_orig_concat.size())
        
        #Then, we are passing it through transformer
        mfn_mem_lstm_input = self.mem_in_drop(self.self_attention_module(all_three_orig_concat,X_pos_Context,Y)).squeeze(0)
        
        #my_logger.debug("Getting output from transformer:",mfn_mem_lstm_input.size())
        
        return mfn_hl_input,mfn_ha_input,mfn_hv_input,mfn_mem_lstm_input
        
        
        
        
        
        

        

class Contextual_MFN(nn.Module):
    def __init__(self,_config,logger):
        super(Contextual_MFN, self).__init__()
        global my_logger
        my_logger = logger
        #my_logger.debug("config in mfn)
        self.config=_config
        #print("the config in mfn_configs:",_config["mfn_configs"][0])
        self.unimodal_context = Unimodal_Context(_config)
        self.multimodal_context = Multimodal_Context(_config)
        self.mfn = MFN(_config)
        
        
        
    def forward(self,X_Punchline,X_Context,X_pos_Context,Y):
        
        #if we don't want context, we will make context all zero here
        if(self.config["use_context"]==False):
                X_Context = torch.zeros_like(X_Context,requires_grad=True)
        
        #Our X_punchline is in format batch_size*time*features.MFN expects it in time*batch_size*features
        #since it performs operation per time index across all batched. So, we will swap axes here.
        #since the concept of "batch" is absent n dataloader get_item, we are doing it here.
        X_Punchline = X_Punchline.permute(1,0,2)
        
        
        text_uni,audio_uni,video_uni = self.unimodal_context.forward(X_Context)
        #print("unimodal complete:",text_uni.shape, audio_uni.shape, video_uni.shape)

        mfn_hl_input,mfn_ha_input,mfn_hv_input,mfn_h_mem_input = \
          self.multimodal_context.forward(text_uni,audio_uni,video_uni,X_pos_Context,Y)
          
        #print("Ready to init the mfn with this:","L:",mfn_hl_input.shape,"A:",mfn_ha_input.shape,\
              #"V:",mfn_hv_input.shape,"mem:",mfn_h_mem_input.shape) 
        prediction = self.mfn.forward(X_Punchline,mfn_hl_input,mfn_ha_input,mfn_hv_input,mfn_h_mem_input)
        #print("result from mfn:",prediction)
        return prediction
        #h_l_prior,h_a_prior,h_v_prior,mem_prior
        
        
       
        

        
        
        
        
