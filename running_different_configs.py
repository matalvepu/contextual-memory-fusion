
from driver import ex
import random
import os
import argparse, sys
import pickle
parser=argparse.ArgumentParser()
parser.add_argument('--dataset', help='the dataset you want to work on')

dataset_specific_config = {
        #Train:10569,dev:2642,Test:3303
        "TED_humor":{'input_dims':[1,81,75,300],'max_seq_len':20,'dev_batch_size':2645,'test_batch_size':3305},
        "TED_humor_albert":{'input_dims':[1,81,75,768],'max_seq_len':20,'dev_batch_size':2645,'test_batch_size':3305},

        "mosi":{'input_dims':[300,5,20],'text_indices':(0,300),'audio_indices':(300,305),'video_indices':(305,325),'max_seq_len':20,'dev_batch_size':250,'test_batch_size':700},
        "iemocap":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "mmmo":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "moud":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "pom":{'text_indices':(0,300),'audio_indices':(300,343),'video_indices':(343,386),'max_seq_len':21},
        "youtube":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21}
        
        }
 # use_context=True
 #    use_context_text=True
 #    use_context_audio=True
 #    use_context_video = True
    
 #    use_punchline_text=True
 #    use_punchline_audio=True
 #    use_punchline_video=True
experiment_configs=[
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':True,'use_context_text':True,'use_context_audio':True,'use_context_video':True},#ind 0:context,T+A+V
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':False,'use_context_text':True,'use_context_audio':False,'use_context_video':False},#ind 1:context,T
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':False,'use_context_text':True,'use_context_audio':True,'use_context_video':False},#ind 2:Context,T+A
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':True,'use_context_text':True,'use_context_audio':False,'use_context_video':True},#ind 3:context, T+V
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':True},#ind 4:No context,T+A+V
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':False},#ind 5: NO context, T
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':False},#ind 6: No context, T+A
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':True},#ind 7: No context, T+V
        {'use_context':True,'use_punchline_text':False,'use_punchline_audio':True,'use_punchline_video':True,'use_context_text':False,'use_context_audio':True,'use_context_video':True},#ind:8,context,A+V
        {'use_context':False,'use_punchline_text':False,'use_punchline_audio':True,'use_punchline_video':True},#ind:9,No Context,A+V
        
        {'use_punchline':False,'use_context_text':True,'use_context_audio':True,'use_context_video':True},#ind:10,No punch,Context:T+A+V
        {'use_punchline':False,'use_context_text':True,'use_context_audio':True,'use_context_video':False},#ind:11,No punch,Context:T+A
        {'use_punchline':False,'use_context_text':True,'use_context_audio':False,'use_context_video':True},#ind:12,No punch,Context:T+V
        {'use_punchline':False,'use_context_text':True,'use_context_audio':False,'use_context_video':False},#ind:13,No punch,Context:T
        {'use_punchline':False,'use_context_text':False,'use_context_audio':True,'use_context_video':True}#ind:14,No punch,Context:A+V
        ]
num_experiments = len(experiment_configs)

#sacred will generate a different random _seed for every experiment
#and we will use that seed to control the randomness emanating from our libraries

node_index=30
#node_index=int(os.environ['SLURM_ARRAY_TASK_ID'])

#So, we are assuming that there will a folder called /processed_multimodal_data in the parent folder
#of this code. I wanted to keep it inside the .git folder. But git push limits file size to be <=100MB
#and some data files exceeds that size.
#all_datasets_location = "../processed_multimodal_data"

#due to limited space, we will directly  use the  data  in mhoque lab
all_datasets_location = "/scratch/mhoque_lab/datasets/processed_multimodal_data/humor/CMFN"


two_context_t_a_v=1
selective_omission=2
discarding_punchline=3


emphaisis_on_a_subset=4
cur_experiment= emphaisis_on_a_subset

best_config = pickle.load(open("best_config_for_zooming.pkl","rb"))
def run_configs(dataset_location):
    dataset_name = dataset_location[dataset_location.rfind("/")+1:]

    if cur_experiment == two_context_t_a_v:
        #print(best_config["unimodal_context"]["hidden_sizes"])
        
                                                     
       
        if node_index<2:
            relevant_config = 0#to ramp up c+T+A+V
            appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[relevant_config],"node_index":node_index,
                                          "prototype":False,'dataset_location':dataset_location,"dataset_name":dataset_name,
                                          "experiment_config_index":relevant_config,'epoch':80}
        else:
            appropriate_config_dict=best_config
            hidden_text =random.choice([32,64,88,128,156,256,512])
            hidden_audio = random.choice([8,16,32,48,64,80,90,100])
            hidden_video = random.choice([8,16,32,48,64,80,90,100])
            appropriate_config_dict["unimodal_context"]["hidden_sizes"] = [hidden_text,hidden_audio,hidden_video]
            #print(appropriate_config_dict["unimodal_context"]["hidden_sizes"])
            appropriate_config_dict["multimodal_context_configs"]['n_source_features'] = sum(appropriate_config_dict["unimodal_context"]["hidden_sizes"])
            appropriate_config_dict["node_index"] = node_index
            
            #appropriate_config_dict["prototype"]=True
            #appropriate_config_dict["epoch"]=2
            
            appropriate_config_dict.pop("device",None)
            #print(appropriate_config_dict)
            #print(appropriate_config_dict.keys())
                    
        
        r= ex.run(config_updates=appropriate_config_dict)
    
    elif cur_experiment==discarding_punchline:
        
        for relevant_config in range(10,15):
            appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[relevant_config],"node_index":node_index,
                                              "prototype":False,'dataset_location':dataset_location,"dataset_name":dataset_name,
                                              "experiment_config_index":relevant_config}
            r= ex.run(config_updates=appropriate_config_dict)
    elif cur_experiment==emphaisis_on_a_subset:
        for relevant_config in [0]:
            appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[relevant_config],"node_index":node_index,
                                              "prototype":True,'dataset_location':dataset_location,"dataset_name":dataset_name,
                                              "experiment_config_index":relevant_config,"epoch":30}
            r= ex.run(config_updates=appropriate_config_dict)

            
    elif cur_experiment==selective_omission:
    #for the selective feature omission, we are choosing context+T+A+V
        relevant_config=0
        num_omit_configs=17
        
        while(True):
    
            for omit_index in range(0,num_omit_configs):
                appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[relevant_config],"node_index":node_index,
                                          "prototype":False,'dataset_location':dataset_location,"dataset_name":dataset_name,
                                          "experiment_config_index":relevant_config,"selectively_omitted_index":omit_index,"omit_corrected":"yes"}
                r= ex.run(config_updates=appropriate_config_dict)
            break
    #print(appropriate_config_dict)
    #Just run it many times
        
    #r = ex.run(named_configs=['search_space'],config_updates={"node_index":node_index,"prototype":True})
    
    
#run it like ./running_different_configs.py --dataset=mosi
#or python running_different_configs.py --dataset=TED_humor

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_path = os.path.join(all_datasets_location,args.dataset)
    
    
    if(os.path.isdir(dataset_path)):
        while(True):
            run_configs(dataset_path)
    else:
        raise NotADirectoryError("Please input the dataset name correctly")
    
    
