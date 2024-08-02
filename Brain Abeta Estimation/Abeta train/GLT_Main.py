import os
import torch
import torch.nn as nn
import nibabel as nib
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
from GlobalLocalTransformer import GlobalAttention,convBlock,GlobalLocalBrainAge
import pandas as pd
import math
import random
import json
import pickle
import cv2
from GLT_train_test import test, train_validate
#discarding because the gray matter map is zeros

max_val=0.15
min_val=0.015

discard=['022_S_6796_0','041_S_6292_0','016_S_6381_0','067_S_6442_0',
         '033_S_6705_0','070_S_4856_1','016_S_6839_0','041_S_6136_0',
         '041_S_6354_0','022_S_6797_0','941_S_6570_0','941_S_6054_0',
         '016_S_6834_0','070_S_4856_0','022_S_6013_0','141_S_6240_0',
         '941_S_6080_0','114_S_6347_0','067_S_6443_0','094_S_6485_0',
         '941_S_6094_0','067_S_6045_0','023_S_6369_0','053_S_6598_0','053_S_6861_0']

root_directory="/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_Global_local_transformer/gray_map_5_slices/"
manual_slice = {
    "axial":list(range(41,60,2)),
    "sagittal":list(range(60,69,2))+list(range(111,120,2)),
    "coronal":list(range(101,120,2))
}
#manual_slice=None
config={
    "lr":5e-5,
    "num_worker":8,
    "epoch":58,
    "batch_size":64,
    "train_test_ratio":0.9,
    "train_val_ratio":0.85,
    "a_model_dir":root_directory+"axial/fold",
    "s_model_dir":root_directory+"sagittal/fold",
    "c_model_dir":root_directory+"coronal/fold",
    "slice":5,
    "patch_size":64,
    "manual_slice":None,
    "excelfile": "/projectnb/bucbi/DL_framework/Regression_data/AB4240_cleaned.xlsx",
    "features":"/projectnb/bucbi/DL_framework/Brainage_multichannel_clean/",
    #"features":"/projectnb/bucbi/DL_framework/Brainage_multichannel_clean/",
    "col_name": "ABETA42/40",
    "root_dir": root_directory,
    "fold_split":"/projectnb/bucbi/DL_framework/Brainage_multichannel_clean/strat_gray_abeta4240.json",
    "pkl_file":"raw_gray_abeta4240.pkl",
    "second_channel_pklname":"gm",
    "discard":discard
}


def test_main(fold,test_loader_a,test_loader_s,test_loader_c,a_model_path,s_model_path,c_model_path,config=None,val=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed=fold
  
    axial_model = GlobalLocalBrainAge(2,patch_size=64,step=32,nblock=6,backbone='vgg8')
    axial_model = nn.DataParallel(axial_model).to(device)
    axial_model.load_state_dict(torch.load(a_model_path))
    predictions_a,targets = test(axial_model,test_loader_a)
    axial_model=None
    
    print("------------------------------------------------------------------------------")
    sagittal_model = GlobalLocalBrainAge(2,patch_size=64,step=32,nblock=6,backbone='vgg8')
    sagittal_model = nn.DataParallel(sagittal_model).to(device)
    sagittal_model.load_state_dict(torch.load(s_model_path))
    predictions_s,_ = test(sagittal_model,test_loader_s)
    sagittal_model=None
    
    print("------------------------------------------------------------------------------")
    coronal_model = GlobalLocalBrainAge(2,patch_size=64,step=32,nblock=6,backbone='vgg8')
    coronal_model = nn.DataParallel(coronal_model).to(device)
    coronal_model.load_state_dict(torch.load(c_model_path))
    predictions_c,_ = test(coronal_model,test_loader_c)
    axial_model=None
    
    predictions_a = torch.FloatTensor(predictions_a)
    predictions_s = torch.FloatTensor(predictions_s)
    predictions_c = torch.FloatTensor(predictions_c)
    sum_3 = torch.stack([predictions_a,predictions_s,predictions_c],dim=0)
    pred_avg = torch.mean(sum_3,dim=0)


    #sorted_ind = [i[0] for i in sorted(enumerate(targets), key=lambda x:x[1])]
    targets,sorted_ind = torch.sort(targets)
    print("Max and min:",torch.max(targets), torch.min(targets))
    criterion = nn.L1Loss()
    pred_avg = np.array([pred_avg[i] for i in sorted_ind])
    pred_avg = np.squeeze(np.average(pred_avg.reshape((-1,config["slice"])),axis=1))
    targets = np.squeeze(np.average(targets.reshape((-1,config["slice"])),axis=1))

    mae = criterion(torch.tensor(pred_avg),torch.tensor(targets))
    
    
    #mae = criterion(pred_avg,targets)
    if val==True:
        print("MAE on validation set with predictions averaging on 3 models:", mae)
    else:
        print("MAE on test set with predictions averaging on 3 models:", mae)
    fig,ax = plt.subplots()
    ax.scatter(targets,pred_avg,vmin=min_val, vmax=max_val)
    cor=np.corrcoef(targets,pred_avg)[0,1]
    plt.suptitle("correlation:"+str(cor))
    plt.xlabel("target "+config["col_name"])
    plt.ylabel("predicted "+config["col_name"])
    plt.title("correlation graph")
    plt.plot([min_val,max_val],[min_val,max_val],'k-',lw=2)
    if val==True:
        plt.title("validation mae on fold"+str(fold)+":"+str(mae))
        plt.savefig(config["root_dir"]+"val_"+str(config["slice"])+"slice_multichannel"+str(fold)+".png")
    else:
        plt.title("test mae on fold"+str(fold)+":"+str(mae))
        plt.savefig(config["root_dir"]+"test_"+str(config["slice"])+"slice_multichannel"+str(fold)+".png")

class MRI(torch.utils.data.Dataset):
    def __init__(self,fold_config,train_val_test,mri_folder, excel_file_path, label_col,slice_number=5, discard = [], show_plane="axial",seed=42,fold=0,manual_slice=None):

        data = pd.read_excel(excel_file_path)
        self.max_val = data[label_col].max()
        self.min_val = data[label_col].min()
        self.target = data
        self.image_folder=mri_folder
        
        self.slice_number=slice_number
        self.show_plane = show_plane
        
        self.file_paths=[]
        f = open(fold_config)
        
        data = json.load(f)
        
        subjects = data[train_val_test][fold]
        
        self.excel_path = excel_file_path
        self.label_col = label_col
        self.show_plane = show_plane
        self.d_slice=[]
        self.h_slice=[]
        self.w_slice=[]
        h,w,d = (182,218,182)
        self.h,self.w,self.d=h,w,d
        if manual_slice==None:
            i,j,k=int(h/3),int(w/3), int(d/3)
            step_h,step_w, step_d = int((h/3)/self.slice_number),int((w/3)/self.slice_number),int((d/3)/self.slice_number)
            while i<((2*h)/3):
                self.h_slice.append(i)
                i+=step_h
            while j<((2*w)/3):
                self.w_slice.append(j)
                j+=step_w
            while k<((2*d)/3):
                self.d_slice.append(k)
                k+=step_d
        else:
            self.d_slice = manual_slice["axial"]
            self.h_slice = manual_slice["sagittal"]
            self.w_slice = manual_slice["coronal"]
        ss=None
        if show_plane=="axial":
            ss = self.d_slice
        if show_plane=="sagittal":
            ss = self.h_slice  
        if show_plane=="coronal":
            ss = self.w_slice
        if manual_slice==None:
            ss=ss[:slice_number]
        for subject in subjects:
            if subject in discard:
                continue
            for s in ss:
                self.file_paths.append((os.path.join(mri_folder, subject),s))
        random.Random(seed).shuffle(self.file_paths)
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self,index):       
        dir_path = self.file_paths[index][0]
        ind = self.file_paths[index][1]
        
        filename = os.path.join(dir_path, config["pkl_file"])
        with open(filename,'rb') as f:
            data = pickle.load(f)
        second_array =  data[config["second_channel_pklname"]]
        raw_array = data["raw"]
        target = data["target"]
        if self.show_plane == "axial":
            raw_array = raw_array[:,:,ind]
            second_array = second_array[:,:,ind]  
        if self.show_plane == "sagittal":
            raw_array = raw_array[ind,:,:]
            second_array = second_array[ind,:,:]
        if self.show_plane == "coronal":
            raw_array = raw_array[:,ind,:]
            second_array = second_array[:,ind,:]
        H,W = raw_array.shape
        h_size = (H//config["patch_size"])*config["patch_size"]
        w_size = (W//config["patch_size"])*config["patch_size"]
        raw_cropped = raw_array[int((H//2)-(h_size/2)):int((H//2)+(h_size/2)),int((W//2)-(w_size/2)):int((W//2)+(w_size/2))]
        sec_cropped = second_array[int((H//2)-(h_size/2)):int((H//2)+(h_size/2)),int((W//2)-(w_size/2)):int((W//2)+(w_size/2))]
        feature = np.dstack((raw_cropped,sec_cropped))
        feature = np.transpose(feature,(2,0,1))
        if math.isnan(target):
            print("error: found nan target. Filename is", filename)
        return feature,target

def fill_locations(model,dataset):
    model.eval()
    a,b = dataset[0]
    model.HW = (a.shape[1],a.shape[2])
    a=np.expand_dims(a, axis=0)
    res = model(torch.Tensor(a))
    model.train()
    return

def get_loader(config,plane,fold):
    d1 = MRI(config["fold_split"],"train",config["features"],config["excelfile"],config["col_name"],
            show_plane = plane,slice_number=config["slice"],discard=config["discard"],fold=fold-1,manual_slice=config["manual_slice"])  
    train_loader = DataLoader(d1,batch_size=config["batch_size"],num_workers=config["num_worker"],shuffle=False)
    d2=MRI(config["fold_split"],"validation",config["features"],config["excelfile"],config["col_name"],
           show_plane = plane,slice_number=config["slice"],discard=config["discard"],fold=fold-1,manual_slice=config["manual_slice"])  
    val_loader = DataLoader(d2,batch_size=config["batch_size"],num_workers=config["num_worker"],shuffle=False)
    d3 = MRI(config["fold_split"],"test",config["features"],config["excelfile"],config["col_name"],
            show_plane = plane,slice_number=config["slice"],discard=config["discard"],fold=fold-1,manual_slice=config["manual_slice"])  
    test_loader = DataLoader(d3,batch_size=config["batch_size"],num_workers=config["num_worker"],shuffle=False)
    return train_loader, val_loader, test_loader,d1,d2,d3

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 
    for fold in [1,2,3]:
        print("graymatter fold:", fold)
        train_loader, val_loader_a,test_loader_a,d_train,d_v,d_test = get_loader(config,"axial",fold)
        save_path_a = config["a_model_dir"]+str(fold)+"/slice"+str(config["slice"])+".pt"
        
        #instantiate model and parallelize in gpu
        model = GlobalLocalBrainAge(2,patch_size=config["patch_size"],step=32,nblock=6,backbone='vgg8')
        fill_locations(model,d_test)
        model = nn.DataParallel(model).to(device)
        #perform training while validating, and save the checkpoint with best val_loss
        train_validate(model,train_loader,val_loader_a,save_path_a,plane="axial",fold=fold,config=config)
        
        model=None
        print("Training the model on sagittal:")
        train_loader, val_loader_s,test_loader_s,d_train,d_v,d_test = get_loader(config,"sagittal",fold)
        
        save_path_s = config["s_model_dir"]+str(fold)+"/slice"+str(config["slice"])+".pt"
        
        model = GlobalLocalBrainAge(2,patch_size=config["patch_size"],step=-1,nblock=6,backbone='vgg8')
        fill_locations(model,d_test)
        model = nn.DataParallel(model).to(device)
        train_validate(model,train_loader,val_loader_s,save_path_s,plane="sagittal",fold=fold,config=config)
        model=None
        print("Training the model on coronal:")
        train_loader, val_loader_c,test_loader_c,d_train,d_v,d_test = get_loader(config,"coronal",fold)
        save_path_c = config["c_model_dir"]+str(fold)+"/slice"+str(config["slice"])+".pt"
        
        model = GlobalLocalBrainAge(2,patch_size=config["patch_size"],step=-1,nblock=6,backbone='vgg8')
        fill_locations(model,d_test)
        model = nn.DataParallel(model).to(device)
        train_validate(model,train_loader,val_loader_c,save_path_c,plane="coronal",fold=fold,config=config)
        #perform current fold testing on test data with preloaded test loader from all 3 planes 
        model=None
        test_main(fold,val_loader_a,val_loader_s,val_loader_c,save_path_a,save_path_s,save_path_c,config=config,val=True)
        test_main(fold,test_loader_a,test_loader_s,test_loader_c,save_path_a,save_path_s,save_path_c,config=config,val=False)
