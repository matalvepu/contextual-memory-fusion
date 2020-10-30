import faulthandler
faulthandler.enable()
import sys
import numpy as np
import random
import torch
import tqdm
import os
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

import h5py
import time
from collections import defaultdict, OrderedDict
import argparse
import pickle
import time
import json, os, ast, h5py
import math

from models import MFN
from models import Contextual_MFN


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from sacred import Experiment
from tqdm import tqdm
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

ex = Experiment('multimodal_humor')
from sacred.observers import MongoObserver

#We must change url to the the bluehive node on which the mongo server is running
url_database = 'bhc0086:27017'
#mongo_database_name = 'real_data_f_score'
#mongo_database_name = 'last_ditch_effort'
#mongo_database_name = 'statistical_test'#in new database reviewer_multi_humor
#mongo_database_name = 'prototype'
#mongo_database_name = 'omitting_punchline'
mongo_database_name = 'albert_embedding'


ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

my_logger = logging.getLogger()
my_logger.disabled=True

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

@ex.config
def cfg():
    node_index = 0
    epoch = 50 #paul did 50
    shuffle = True
    num_workers = 2
    best_model_path =  "/scratch/echowdh2/saved_models_from_projects/multimodal_humor/"+str(node_index) +"_best_model.chkpt"
    num_context_sequence=5
    experiment_config_index=0
    
    dataset_location = None
    dataset_name = None
    text_indices = None
    audio_indices=None
    video_indices = None
    max_seq_len = None
    input_dims=None #organized as [t,a,v]
    

    
    padding_value = 0.0
    
    #This variable denotes which feature we are selecting to remove from dataset
    #It will be passed from running_different_configs.py
    selectively_omitted_index=-1
    omit_corrected=None
    #Each entry of this list will contain a dict about which feature we are omitting.
    #It will have feature name, associated modality 
    #We will index into this dict based on the valud of the last variable
 

    selective_audio_visual_feature_omission=[
            {"modality":"video","name":"happiness","indices": [44, 48, 61, 65]},
            {"modality":"video","name":"sadness","indices": [40, 42, 50, 57, 59, 67]},
            {"modality":"video","name":"surprise","indices": [40, 41, 43, 55, 57, 58, 60, 72]},
            {"modality":"video","name":"anger","indices": [42, 43, 45, 53, 59, 60, 62, 70]},
            {"modality":"video","name":"disgust","indices": [46, 50, 51, 63, 67, 68]},
            {"modality":"video","name":"fear","indices": [40, 41, 42, 43, 45, 52, 55, 57, 58, 59, 60, 62, 69, 72]},


            {"modality":"video","name":"upper_face","indices": [40, 41, 42, 43, 44, 45, 56, 57, 58, 59, 60, 61, 62, 74]},
            {"modality":"video","name":"lower_face","indices": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]},
            {"modality":"video","name":"shape_params","indices":  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]},
            {"modality":"audio","name":"pitch","indices": [0]},
            {"modality":"audio","name":"harmonic","indices": [4]},
            {"modality":"audio","name":"quotient","indices": [3]},
            {"modality":"audio","name":"mcep","indices":  [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]},
            {"modality":"audio","name":"pdm","indices":  [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]},
            {"modality":"audio","name":"pdd","indices":  [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]},
            {"modality":"audio","name":"formant","indices": [75, 76, 77, 78, 79]},
            {"modality":"audio","name":"peak_slope","indices": [8]}
            ]
            
    #This variable is to keep track of the experiments in omniboard. It will be none at first.
    #But based on the value of selectively_omitted_index, we will assign it properly through config_updates
    omitted_feature_name=None
            
    
    #To ensure that it captures the whole batch at the same time
    #and hence we get same score as Paul
    #TODO: Must cahange
    train_batch_size = random.choice([64,128,256,512])
    #These two are coming from running_different_configs.py
    dev_batch_size=None
    test_batch_size=None
    
    

   
    
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    use_context=True
    use_context_text=True
    use_context_audio=True
    use_context_video = True
    
    use_punchline=True
    use_punchline_text=True
    use_punchline_audio=True
    use_punchline_video=True
    
    
    
    
    save_model = "best_model"
    save_mode = 'best'
    
    prototype=False
    if prototype:
        epoch=1
        
    #TODO: May have to change the hidden_sizes to match with later stages
    #TODO:Will need to add RANDOM CHOICE FOR hidden_size later
    #Basically the hidden_sizes is an arry containing hidden_size for all four [t_old,a,v,t_embedded]
    #THe LSTM will get the embedded text vector,so we are using the last one 
    hidden_text =random.choice([32,64,88,128,156,256])
    hidden_audio = random.choice([8,16,32,48,64,80])
    hidden_video = random.choice([8,16,32,48,64,80])
    unimodal_context = {"text_lstm_input":input_dims[3],"audio_lstm_input":input_dims[1],
                        "video_lstm_input":input_dims[2],"hidden_sizes":[hidden_text,hidden_audio,hidden_video]
                                                     }
    
    multimodal_context_configs = {'d_word_vec':512,'d_model':512,'d_inner_hid':2048,
                   'd_k':64,'d_v':64,'n_head':8,'n_layers':6,'n_warmup_steps':4000,
                   'dropout':0.1,'embs_share_weight':True,'proj_share_weight':True,
                   'label_smoothing': True,'max_token_seq_len':num_context_sequence,
                   'n_source_features':sum(unimodal_context["hidden_sizes"]),
                   'text_in_drop':random.choice([0.0,0.1, 0.2,0.5]),
                   'audio_in_drop':random.choice([0.0,0.2,0.5,0.1]),
                   'video_in_drop':random.choice([0.0,0.2,0.5,0.1]),
                   'mem_in_drop':random.choice([0.0,0.2,0.5,0.1])
                   
                   }

        
    #All these are mfn configs    
    config = dict()
    config["input_dims"] = input_dims
    hl = random.choice([32,64,88,128,156,256])
    ha = random.choice([8,16,32,48,64,80])
    hv = random.choice([8,16,32,48,64,80])
    config["h_dims"] = [hl,ha,hv]
    config["memsize"] = random.choice([64,128,256,300,400])
    config["windowsize"] = 2
    config["batchsize"] = random.choice([32,64,128,256])
    config["num_epochs"] = 50
    config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01])
    config["momentum"] = random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
    
    NN1Config = dict()
    NN1Config["shapes"] = random.choice([32,64,128,256])
    NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    NN2Config = dict()
    NN2Config["shapes"] = random.choice([32,64,128,256])
    NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    gamma1Config = dict()
    gamma1Config["shapes"] = random.choice([32,64,128,256])
    gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    gamma2Config = dict()
    gamma2Config["shapes"] = random.choice([32,64,128,256])
    gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    outConfig = dict()
    outConfig["shapes"] = random.choice([32,64,128,256])
    outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    mfn_configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
   
class HumorDataset(Dataset):
    
    def __init__(self,id_list,_config,):
        self.id_list = id_list
        data_path = _config["dataset_location"]
            


        openface_file= os.path.join(data_path,'word_aligned_openface_sdk.pkl')
        covarep_file=os.path.join(data_path,"word_aligned_covarep_sdk.pkl")
        word_idx_file=os.path.join(data_path,"humor_word_embedding_idx_sdk.pkl")
        self.word_aligned_openface_sdk=load_pickle(openface_file)
        self.word_aligned_covarep_sdk=load_pickle(covarep_file)
        self.word_embedding_idx_sdk=load_pickle(word_idx_file)
        self.of_d=75
        self.cvp_d=81
        self.max_context_len=_config["num_context_sequence"]
        self.max_sen_len=_config["max_seq_len"]
    
    def paded_word_idx(self,seq,max_sen_len=20,left_pad=1):
        seq=seq[0:max_sen_len]
        pad_w=np.concatenate((np.zeros(max_sen_len-len(seq)),seq),axis=0)
        pad_w=np.array([[w_id] for  w_id in pad_w])
        return pad_w

    def padded_covarep_features(self,seq,max_sen_len=20,left_pad=1):
        seq=seq[0:max_sen_len]
        return np.concatenate((np.zeros((max_sen_len-len(seq),self.cvp_d)),seq),axis=0)

    def padded_openface_features(self,seq,max_sen_len=20,left_pad=1):
        seq=seq[0:max_sen_len]
        return np.concatenate((np.zeros(((max_sen_len-len(seq)),self.of_d)),seq),axis=0)

    def padded_context_features(self,context_w,context_of,context_cvp,max_context_len=5,max_sen_len=20):
        context_w=context_w[-max_context_len:]
        context_of=context_of[-max_context_len:]
        context_cvp=context_cvp[-max_context_len:]

        padded_context=[]
        for i in range(len(context_w)):
            p_seq_w=self.paded_word_idx(context_w[i],max_sen_len)
            p_seq_cvp=self.padded_covarep_features(context_cvp[i],max_sen_len)
            p_seq_of=self. padded_openface_features(context_of[i],max_sen_len)
            padded_context.append(np.concatenate((p_seq_w,p_seq_cvp,p_seq_of),axis=1))

        pad_c_len=max_context_len-len(padded_context)
        padded_context=np.array(padded_context)
        
        if not padded_context.any():
            return np.zeros((max_context_len,max_sen_len,157))
        
        return np.concatenate((np.zeros((pad_c_len,max_sen_len,157)),padded_context),axis=0)
    
    def padded_punchline_features(self,punchline_w,punchline_of,punchline_cvp,max_sen_len=20,left_pad=1):
        
        p_seq_w=self.paded_word_idx(punchline_w,max_sen_len)
        p_seq_cvp=self.padded_covarep_features(punchline_cvp,max_sen_len)
        p_seq_of=self.padded_openface_features(punchline_of,max_sen_len)
        return np.concatenate((p_seq_w,p_seq_cvp,p_seq_of),axis=1)
        
    
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self,index):
        
        hid=self.id_list[index]
        punchline_w=np.array(self.word_embedding_idx_sdk[hid]['punchline_emb_idx_list'])
        punchline_of=np.array(self.word_aligned_openface_sdk[hid]['punchline_features'])
        punchline_cvp=np.array(self.word_aligned_covarep_sdk[hid]['punchline_features'])
        
        context_w=np.array(self.word_embedding_idx_sdk[hid]['context_embd_idx_matrix'])
        context_of=np.array(self.word_aligned_openface_sdk[hid]['context_features'])
        context_cvp=np.array(self.word_aligned_covarep_sdk[hid]['context_features'])
        
        
        X_punch=torch.FloatTensor(self.padded_punchline_features(punchline_w,punchline_of,punchline_cvp,self.max_sen_len))
        
        X_context=torch.FloatTensor(self.padded_context_features(context_w,context_of,context_cvp,self.max_context_len,self.max_sen_len))
        #Basically, we will think the whole sentence as a sequence.
        #all the words will be merged. If all of them are zero, then it is a padding 
        reshaped_context = torch.reshape(X_context,(X_context.shape[0],-1))
        #my_logger.debug("The reshaped context:",reshaped_context.size())
        padding_rows = np.where(~reshaped_context.cpu().numpy().any(axis=1))[0]
        n_rem_entries= reshaped_context.shape[0] - len(padding_rows)
        X_pos_context = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        #my_logger.debug("X_pos:",X_pos," Len:",X_pos.shape)
        X_pos_context = torch.LongTensor(X_pos_context)   
        #my_logger.debug("X_pos_context:",X_pos_context.shape,X_pos_context)
        
        Y=torch.FloatTensor([self.word_embedding_idx_sdk[hid]['label']])
                
        return X_punch,X_context,X_pos_context,Y

class Generic_Dataset(Dataset):
    def __init__(self, X, Y,_config):
        self.X = X
        self.Y = Y
        self.config = _config
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        
        #my_logger.debug("The P:",X.size(),X[:,:])
        #TODO: Must change it when correct dataset arrives
        #we are just repeating each entry 
        X_context = torch.FloatTensor(np.repeat(self.X[idx],
            self.config["num_context_sequence"],0).reshape((self.config["num_context_sequence"],
                       self.config["max_seq_len"],-1))) 
        #my_logger.debug("The Context:",X_context.size())
        
        #Basically, we will think the whole sentence as a sequence.
        #all the words will be merged. If all of them are zero, then it is a padding 
        reshaped_context = torch.reshape(X_context,(X_context.shape[0],-1))
        #my_logger.debug("The reshaped context:",reshaped_context.size())
        padding_rows = np.where(~reshaped_context.cpu().numpy().any(axis=1))[0]
        n_rem_entries= reshaped_context.shape[0] - len(padding_rows)
        X_pos_context = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        #my_logger.debug("X_pos:",X_pos," Len:",X_pos.shape)
        X_pos_context = torch.LongTensor(X_pos_context)   
        #my_logger.debug("X_pos_context:",X_pos_context.shape,X_pos_context)

        Y = torch.FloatTensor([self.Y[idx]])
        #TODO:MUST ERASE in new dataset, doing to run sigmoid
        Y = Y>0
        #my_logger.debug("The new Y:",Y)
        
        return X,X_context,X_pos_context,Y




@ex.capture        
def load_saved_data(_config):
    
    data_path = os.path.join(_config["dataset_location"],'data')
    #TODO:Change it properly
    
    h5f = h5py.File(os.path.join(data_path,'X_train.h5'),'r')
    X_train = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_train.h5'),'r')
    y_train = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'X_valid.h5'),'r')
    X_valid = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_valid.h5'),'r')
    y_valid = h5f['data'][:]
    h5f.close()
    
    h5f = h5py.File(os.path.join(data_path,'X_test.h5'),'r')
    X_test = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_test.h5'),'r')
    y_test = h5f['data'][:]
    h5f.close()
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

@ex.capture
def set_up_data_loader(_config):
    # train_X,train_Y,dev_X,dev_Y,test_X,test_Y = load_saved_data()
    # my_logger.debug("all data loaded. Now creating data loader")
    
    # if(_config["prototype"]):
    #     train_X = train_X[:10,:,:]
    #     train_Y = train_Y[:10]
        
    #     dev_X = dev_X[:10,:,:]
    #     dev_Y = dev_Y[:10]
        
    #     test_X = test_X[:10,:,:]
    #     test_Y = test_Y[:10]
    dataset_id_file= os.path.join(_config["dataset_location"], "dataset_id_sdk.pkl")
    dataset_id=load_pickle(dataset_id_file)
    train=dataset_id['train']
    dev=dataset_id['dev']
    test=dataset_id['test']
    if(_config["prototype"]):
        train=train[:10]
        dev=dev[:10]
        test=test[:10]
    training_set = HumorDataset(train,_config)
    dev_set = HumorDataset(dev,_config)
    test_set = HumorDataset(test,_config)
    # train_dataset = Generic_Dataset(train_X,train_Y,_config = _config)
    # dev_dataset = Generic_Dataset(dev_X,dev_Y,_config=_config)
    # test_dataset = Generic_Dataset(test_X,test_Y,_config=_config)
    
    train_dataloader = DataLoader(training_set, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    dev_dataloader = DataLoader(dev_set, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    test_dataloader = DataLoader(test_set, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    
    #my_logger.debug(train_X.shape,train_Y.shape,dev_X.shape,dev_Y.shape,test_X.shape,test_Y.shape)
    #data_loader = test_data_loader(train_X,train_Y,_config)
    return train_dataloader,dev_dataloader,test_dataloader

@ex.capture
def set_random_seed(_seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)



@ex.capture
def train_epoch(model, training_data, criterion,optimizer, device, smoothing,_config):
    ''' Epoch operation in training phase'''

    model.train()

    epoch_loss = 0.0
    num_batches = 0
   
    for batch in tqdm(training_data, mininterval=2,desc='  - (Training)   ', leave=False):

     #TODO: For simplicity, we are not using X_pos right now as we really do not know
     #how it can be used properly. So, we will just use the context information only.
        X_Punchline,X_Context,X_pos_Context,Y = map(lambda x: x.to(device), batch)
        
        
        
        # print("\nData_size:\nX_P:", X_Punchline.shape,", X_C:",X_Context.shape,",X_C_pos:",\
        #       X_pos_Context.shape,"Y:",Y.shape)
                
        # forward
        optimizer.zero_grad()
        predictions = model(X_Punchline,X_Context,X_pos_Context,Y).squeeze(0)
        #my_logger.debug(predictions.size(),train_Y.size())

        loss = criterion(predictions,Y.float())
        loss.backward()
        #optimizer.step()
        epoch_loss += loss.item()

        # update parameters
        #using best mfn config now
        optimizer.step_and_update_lr()
        
        num_batches +=1

    #TODO: MUST REMOVE
    if(num_batches==0):
        num_batches+=1
    return epoch_loss / num_batches

@ex.capture
def eval_epoch(model,data_loader,criterion, device,_config):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
            
            X_Punchline,X_Context,X_pos_Context,Y = map(lambda x: x.to(device), batch)
        
            
         
            
            predictions = model(X_Punchline,X_Context,X_pos_Context,Y).squeeze(0)
            loss = criterion(predictions, Y.float())
            
            epoch_loss += loss.item()
            
            num_batches +=1
    return epoch_loss / num_batches
@ex.capture
def reload_model_from_file(file_path):
        checkpoint = torch.load(file_path)
        _config = checkpoint['_config']
        
        #encoder_config = _config["multimodal_context_configs"]
        model = Contextual_MFN(_config,my_logger).to(_config["device"])
        

        

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        return model
        
@ex.capture
def test_epoch(model,data_loader,criterion, device,_config):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    returned_Y = None
    returned_predictions = None
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
    
         
            X_Punchline,X_Context,X_pos_Context,Y = map(lambda x: x.to(device), batch)
            
           
            predictions = model(X_Punchline,X_Context,X_pos_Context,Y).squeeze(0)
            loss = criterion(predictions, Y.float())
            
            epoch_loss += loss.item()
            
            num_batches +=1
            #if we don'e do the squeeze, it remains as 2d numpy arraya nd hence
            #creates problems like nan while computing various statistics on them
            returned_Y = Y.squeeze(1).cpu().numpy()
            returned_predictions = predictions.squeeze(1).cpu().data.numpy()
            
    return returned_predictions,returned_Y   


    
            
@ex.capture
def train(model, training_data, validation_data, optimizer,criterion,_config,_run):
    ''' Start training '''
    model_path = _config["best_model_path"]

    valid_losses = []
    for epoch_i in range(_config["epoch"]):
        
        train_loss = train_epoch(
            model, training_data, criterion,optimizer, device = _config["device"],
                smoothing=_config["multimodal_context_configs"]["label_smoothing"])
        _run.log_scalar("training.loss", train_loss, epoch_i)


        valid_loss = eval_epoch(model, validation_data, criterion,device=_config["device"])
        _run.log_scalar("dev.loss", valid_loss, epoch_i)
        
        #scheduler.step(valid_loss)

        
        
        valid_losses.append(valid_loss)
        print("\nepoch:{},train_loss:{}, valid_loss:{}".format(epoch_i,train_loss,valid_loss))
      #Due to space3 constraint, we are not saving the models. There should be enough info
      #in sacred to reproduce the results on the fly
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            '_config': _config,
            'epoch': epoch_i}

        if _config["save_model"]:
            if _config["save_mode"] == 'best':
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_path)
                    my_logger.debug('    - [Info] The checkpoint file has been updated.')
    #After the entire training is over, save the best model as artifact in the mongodb, only if it is not protptype
    #Due to space constraint, we are not saving the model since it is not necessary as we know the seed. If we need to regenrate the result
    #simple running it again should work
    # if(_config["prototype"]==False):
    #     ex.add_artifact(model_path)


@ex.capture
def test_score_from_file(test_data_loader,criterion,_config,_run):
    model_path =  _config["best_model_path"]
    model = reload_model_from_file(model_path)

    predictions,y_test = test_epoch(model,test_data_loader,criterion,_config["device"])
    my_logger.debug("predictions:",predictions,predictions.shape)
    my_logger.debug("ytest:",y_test,y_test.shape)
    mae = np.mean(np.absolute(predictions-y_test))
    my_logger.debug("mae: ", mae)
    
    corr = np.corrcoef(predictions,y_test)[0][1]
    my_logger.debug("corr: ", corr)
    
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    #print("mult_acc: ", mult)
    predicted_label = (predictions >= 0)
    f_score = round(f1_score(np.round(predicted_label),np.round(y_test),average='weighted'),5)
    ex.log_scalar("test.f_score",f_score)

    #print("mult f_score: ", f_score)
    
    #TODO:Make sure that it is correct
    #true_label = (y_test >= 0)
    true_label = (y_test)

    

    #print("Confusion Matrix :")
    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    #print(confusion_matrix_result)
    
    #print("Classification Report :")
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    #print(classification_report_score)
    
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy:",accuracy )
    
    _run.info['final_result']={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_acc":mult,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}
    return accuracy

@ex.capture
def test_omit(_config):
    print(_config["selectively_omitted_index"],_config["omitted_feature_name"])


@ex.automain
def driver(_config,_run):
    
    if (_config["selectively_omitted_index"] != -1):
        ex.add_config({"omitted_feature_name":_config["selective_audio_visual_feature_omission"][_config["selectively_omitted_index"]]["name"]})
   
    output = open('config_file.pkl', 'wb')
    pickle.dump(_config, output)
    ex.add_artifact('config_file.pkl')
    output.close()
    
        
    set_random_seed()
    #print("inside driver")
    #X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data()
    #print(X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_data_loader,dev_data_loader,test_data_loader = set_up_data_loader()
    
    
    
    multimodal_context_config = _config["multimodal_context_configs"]
    
    model = Contextual_MFN(_config,my_logger).to(_config["device"])
    #for now, we will use the same scheduler for the entire model.
    #Later, if necessary, we may use the default optimizer of MFN
    #TODO: May have to use separate scheduler for transformer and mfn
    #We are using the optimizer and scgheduler of mfn as a last resort
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        multimodal_context_config["d_model"], multimodal_context_config["n_warmup_steps"])
    
    #TODO: May have to change the criterion
    #criterion = nn.L1Loss()
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(_config["device"])
    # optimizer =  optim.Adam(
    #         filter(lambda x: x.requires_grad, model.parameters()),lr = _config["config"]["lr"],
    #         betas=(0.9, 0.98), eps=1e-09)
    # #torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # #optimizer = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=False)
    # scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)

    train(model, train_data_loader,dev_data_loader, optimizer, criterion)

    #test_accuracy =  test_score_from_model(model,test_data_loader,criterion)
    
    test_accuracy = test_score_from_file(test_data_loader,criterion)
    ex.log_scalar("test.accuracy",test_accuracy)
    results = dict()
    #I believe that it will try to minimize the rest. Let's see how it plays out
    results["optimization_target"] = 1 - test_accuracy
    
    stat_file = open("all_accuracies_for_stat.txt","a") 
 
    stat_file.write(str(_config["experiment_config_index"]) + "," + str(test_accuracy) + "\n") 
    stat_file.close()
    return results

