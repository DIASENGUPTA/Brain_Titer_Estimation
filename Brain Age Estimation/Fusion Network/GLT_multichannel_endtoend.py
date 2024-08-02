import os
import torch
print("-------")
import torch.nn as nn
import nibabel as nib
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
print("-------")
import numpy as np
from torch.utils.data import Dataset,DataLoader
print("-------")
from GlobalLocalTransformer import GlobalAttention,convBlock,GlobalLocalBrainAge
import pandas as pd
import math
import random
import json
#discarding because the gray matter map is zeros
discard=['022_S_6796_0','041_S_6292_0','016_S_6381_0','067_S_6442_0',
         '033_S_6705_0','070_S_4856_1','016_S_6839_0','041_S_6136_0',
         '041_S_6354_0','022_S_6797_0','941_S_6570_0','941_S_6054_0',
         '016_S_6834_0','070_S_4856_0','022_S_6013_0','141_S_6240_0',
         '941_S_6080_0','114_S_6347_0','067_S_6443_0','094_S_6485_0',
         '941_S_6094_0','067_S_6045_0','023_S_6369_0','053_S_6598_0',
         '053_S_6598_0','053_S_6861_0']

config={
    "num_worker":2,
    "k_fold":3,
    "epoch":80,
    "batch_size":32,
    "train_test_ratio":0.9,
    "train_val_ratio":0.85,
    "a_model_dir":"/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_GLT_modified/multichannel_endtoend/axial/fold",
    "s_model_dir":"/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_GLT_modified/multichannel_endtoend/sagittal/fold",
    "c_model_dir":"/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_GLT_modified/multichannel_endtoend/coronal/fold",
    "a_plot_dir":"/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_GLT_modified/multichannel_endtoend/axial/fold",
    "s_plot_dir":"/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_GLT_modified/multichannel_endtoend/sagittal/fold",
    "c_plot_dir":"/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_GLT_modified/multichannel_endtoend/coronal/fold",
    "slice":20,
    "excelfile": "/projectnb/bucbi/DL_framework/Regression_data/AB4240_cleaned.xlsx",
    "features":"/projectnb/bucbi/DL_framework/Brainage_multichannel_clean/",
    "col_name": "AGE",
    "root_dir": "/projectnb/bucbi/DL_framework-Kangxian/ATAB/ATAB_GLT_modified/multichannel_endtoend/"
}

def test(model,val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_preds=[]
    val_targets=[]
    model.eval()
    for i_batch, (data, age) in enumerate(val_dataloader):
        data, age = data.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
        val_targets.append(age[0].item())
        zlist = model(data)
        with torch.no_grad():
            age=torch.unsqueeze(age,1)
            #p=torch.mean(zlist).item()
        val_preds.append(torch.squeeze(zlist).item())
    return val_preds,torch.FloatTensor(val_targets)

def test_main(fold,test_loader_a,test_loader_s,test_loader_c,a_model_path,s_model_path,c_model_path,config=None,val=False,spatial_dim=(182,218,182)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed=fold
    axial_model = GlobalLocalBrainAge(config["slice"]*2,patch_size=64,step=32,nblock=6,backbone='vgg8',spatial_size=(spatial_dim[0],spatial_dim[1]))
    axial_model = nn.DataParallel(axial_model).to(device)
    axial_model.load_state_dict(torch.load(a_model_path))
    predictions_a,targets = test(axial_model,test_loader_a)
    axial_model=None
    
    print("------------------------------------------------------------------------------")
    sagittal_model = GlobalLocalBrainAge(config["slice"]*2,patch_size=64,step=32,nblock=6,backbone='vgg8',spatial_size=(spatial_dim[1],spatial_dim[2]))
    sagittal_model = nn.DataParallel(sagittal_model).to(device)
    sagittal_model.load_state_dict(torch.load(s_model_path))
    predictions_s,_ = test(sagittal_model,test_loader_s)
    sagittal_model=None
    
    print("------------------------------------------------------------------------------")
    coronal_model = GlobalLocalBrainAge(config["slice"]*2,patch_size=64,step=32,nblock=6,backbone='vgg8',spatial_size=(spatial_dim[0],spatial_dim[2]))
    coronal_model = nn.DataParallel(coronal_model).to(device)
    coronal_model.load_state_dict(torch.load(c_model_path))
    predictions_c,_ = test(coronal_model,test_loader_c)
    axial_model=None
    
    predictions_a = torch.FloatTensor(predictions_a)
    predictions_s = torch.FloatTensor(predictions_s)
    predictions_c = torch.FloatTensor(predictions_c)
    sum_3 = torch.stack([predictions_a,predictions_s,predictions_c],dim=0)
    pred_avg = torch.mean(sum_3,dim=0)
    
    criterion = nn.L1Loss()
    mae = criterion(pred_avg,targets)
    if val==True:
        print("MAE on validation set with predictions averaging on 3 models:", mae)
    else:
        print("MAE on test set with predictions averaging on 3 models:", mae)
    fig,ax = plt.subplots()
    ax.scatter(targets,pred_avg,vmin=50, vmax=96)
    cor=np.corrcoef(targets,pred_avg)[0,1]
    plt.suptitle("correlation:"+str(cor))
    plt.xlabel("target age")
    plt.ylabel("predicted_age")
    plt.title("correlation graph")
    plt.plot([53,96],[53,92],'k-',lw=2)
    if val==True:
        plt.title("validation mae on fold"+str(fold)+":"+str(mae))
        plt.savefig(config["root_dir"]+"val_"+str(config["slice"])+"slice_multichannel"+str(fold)+".png")
    else:
        plt.title("test mae on fold"+str(fold)+":"+str(mae))
        plt.savefig(config["root_dir"]+"test_"+str(config["slice"])+"slice_multichannel"+str(fold)+".png")

class MRI(torch.utils.data.Dataset):
    def __init__(self,mri_folder, excel_file_path, label_col,slice_number=5, discard = [], show_plane="axial",seed=42):
        self.numR=0
        self.numW=0
        self.o=[]
        data = pd.read_excel(excel_file_path)
        self.max_val = data[label_col].max()
        self.min_val = data[label_col].min()
        self.target = data
        self.image_folder=mri_folder
        
        self.slice_number=slice_number
        self.show_plane = show_plane
        
        self.file_paths=[]
        for path, currentDirectory, files in os.walk(mri_folder):
      
            for file in files:
                bn = os.path.basename(path)
                if bn in discard:
                    break
                self.file_paths.append(path)
                break
        self.excel_path = excel_file_path
        self.label_col = label_col
        self.d_slice=[]
        self.h_slice=[]
        self.w_slice=[]
        self.show_plane = show_plane
        random.Random(seed).shuffle(self.file_paths)
        h,w,d = (182,218,182)
        self.h=h
        self.w=w
        self.d=d  
        i,j,k=int(h/5),int(w/5), int(d/5)
        step_h,step_w, step_d = int(h*(3/5)/self.slice_number),int(w*(3/5)/self.slice_number),int(d*(3/5)/self.slice_number)
        while i<((4*h)/5):
            self.h_slice.append(i)
            i+=step_h
        while j<((4*w)/5):
            self.w_slice.append(j)
            j+=step_w
        while k<((4*d)/5):
            self.d_slice.append(k)
            k+=step_d
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self,index):       
        
        dir_path = self.file_paths[index]
        folder_name = os.path.basename(dir_path)
        subdata = self.target.loc[self.target['PTID'] == folder_name]
        value = subdata.iloc[0][self.label_col]
        GM_file=None
        WM_file=None
        raw_file=None
        for path, currentDirectory, files in os.walk(dir_path):
            for file in files:
                file_paths = os.path.join(path, file)
                if "GM" in file_paths:
                    GM_file = file_paths
                elif "WM" in file_paths:
                    WM_file = file_paths
                elif "raw" in file_paths:
                    raw_file = file_paths
        
        
        g_img = nib.load(GM_file)
        g_array = g_img.get_fdata()
        g_array = np.asarray(g_array, np.int64)
        r_img = nib.load(raw_file)
        r_array = r_img.get_fdata()
        r_array = np.asarray(r_array, np.int64)
        
        h,w,d = g_array.shape
        
        #w_img = nib.load(WM_file)
        #w_array = w_img.get_fdata()
        #w_array = np.asarray(w_array, np.int64)
        #maxW = np.amax(w_array)
        maxR = np.amax(r_array)
        maxG = np.amax(g_array)
        #minV = np.amin(img_array)
        #img_array = (img_array-minV)/(maxV-minV)
        r_array = (r_array-np.mean(r_array))/np.std(r_array)
        #g_array = (g_array-np.mean(g_array))/np.std(g_array)
        if self.show_plane == "axial":
            g_array = g_array[:,:,self.d_slice[:self.slice_number]]
            r_array = r_array[:,:,self.d_slice[:self.slice_number]]
            g_array = np.transpose(g_array,(2,0,1))
            r_array = np.transpose(r_array,(2,0,1))
        if self.show_plane == "sagittal":
            g_array = g_array[self.w_slice[:self.slice_number],:,:]
            r_array = r_array[self.w_slice[:self.slice_number],:,:]
        if self.show_plane == "coronal":
            g_array = g_array[:,self.h_slice[:self.slice_number],:]
            r_array = r_array[:,self.h_slice[:self.slice_number],:]
            g_array = np.transpose(g_array,(1,0,2))
            r_array = np.transpose(r_array,(1,0,2))
        feature = np.concatenate((r_array,g_array),axis=0)
        return feature,value

def train_validate(model,train_dataloader,val_dataloader,save_path,plane="axial",fold=0,config=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 )
    scheduler = lr_scheduler.StepLR(optimizer,step_size=25,gamma=0.5,verbose=True)
    criterion = nn.L1Loss()
    best_val_loss=1e+10
    val_losses=[]
    training_losses=[]
    fig,ax = plt.subplots()
    for i in range(config["epoch"]):
        running_loss=[]
        model.train()
        print("epoch", i,":")
        for i_batch, (data, age) in enumerate(train_dataloader):
            data, age = data.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
            #print("batch ", i_batch)
            zlist = model(data)
            zlist = torch.squeeze(zlist)
            loss = criterion(zlist,age)
            
            running_loss.append(loss.item())
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss=sum(running_loss)/len(running_loss)
        print("Training loss:",train_loss)
        training_losses.append(train_loss)
        avg_val_loss=[]
        val_preds=[]
        val_targets=[]
        model.eval()
        for i_batch, (data, age) in enumerate(val_dataloader):
            data, age = data.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
            val_targets.append(age[0].item())
            if i_batch==5:
                print(model.module.final.weight)
            zlist = model(data)
            with torch.no_grad():
                age=torch.unsqueeze(age,1)
                val_preds.append(torch.squeeze(zlist).item())
                val_loss = criterion(zlist,age)
            avg_val_loss.append(val_loss.item())
        scheduler.step()
        sorted_ind = [i[0] for i in sorted(enumerate(val_targets), key=lambda x:x[1])]
        val_targets.sort()
        val_preds = [val_preds[i] for i in sorted_ind]
        val_loss_curr = sum(avg_val_loss)/len(avg_val_loss)
        plt.cla()
        ax.scatter(val_targets,val_preds,vmin=50, vmax=96)
        plt.plot([53,92],[53,92],'k-',lw=2)
        cor=np.corrcoef(val_targets,val_preds)[0,1]
        plt.suptitle("pearson correlation:"+str(cor))
        plt.xlabel("target age")
        plt.ylabel("predicted_age")
        plt.title("validation mae loss:"+str(val_loss_curr))
        plt.savefig(config["root_dir"]+plane+"/fold"+str(fold)+"/plots/validation_epoch"+str(i)+".png")
        plt.cla()
        print("validation mae loss:", val_loss_curr)
        if val_loss_curr<6:
            model.module.final.weight.requires_grad = True
        val_losses.append(val_loss_curr)
        if val_loss_curr<best_val_loss:
            torch.save(model.state_dict(), save_path)
            best_val_loss = val_loss_curr
    plt.cla()
    plt.ylim(0,25)
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.plot(val_losses,label='validation_losses')
    plt.plot(training_losses,label='training_losses')
    plt.legend()
    plt.savefig(config["root_dir"]+plane+"/fold"+str(fold)+"/plots/val_loss_epoch.png")
    plt.cla()
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    for fold in [1]:
        seed=(fold+2)*2
        d = MRI(config["features"],config["excelfile"],config["col_name"],
            show_plane = "axial",slice_number=config["slice"],discard=discard)  
        a,v = d[0]
        H,W = a.shape[1],a.shape[2]
        train_len = int(len(d)*0.9)
        test_len = len(d)-train_len
        val_len = int(train_len*(1-0.9))
        train_len = train_len-val_len
        save_path_a = config["a_model_dir"]+str(fold)+"/slice"+str(config["slice"])+".pt"
        #seed a fixed seed for each fold
        torch.manual_seed(seed)
        #randomly split the dataset, same splits within the same fold
        train_set, val_set,test_set = torch.utils.data.random_split(d, [train_len,val_len,test_len])
        #data_loader setup
        train_loader = DataLoader(train_set,batch_size=config["batch_size"],num_workers=config["num_worker"],shuffle=False)
        val_loader_a = DataLoader(val_set,batch_size=1, shuffle=False,num_workers=config["num_worker"])
        test_loader_a = DataLoader(test_set,batch_size=1, shuffle=False,num_workers=config["num_worker"])
        #instantiate model and parallelize in gpu
        model = GlobalLocalBrainAge(config["slice"]*2,patch_size=64,step=32,nblock=6,backbone='vgg8',spatial_size = (H,W))
        model = nn.DataParallel(model).to(device)
        #perform training while validating, and save the checkpoint with best val_loss
        train_validate(model,train_loader,val_loader_a,save_path_a,plane="axial",fold=fold,config=config)
        print("---------------------------------------------")
        model=None
        print("Training the model on sagittal:")
        torch.manual_seed(seed)
        d = MRI(config["features"],config["excelfile"],config["col_name"],
                slice_number=config["slice"],show_plane = "sagittal",discard=discard)  
        a,v = d[0]
        H,W = a.shape[1],a.shape[2]
        save_path_s = config["s_model_dir"]+str(fold)+"/slice"+str(config["slice"])+".pt"
        train_set, val_set, test_set = torch.utils.data.random_split(d, [train_len,val_len,test_len])
        train_loader = DataLoader(train_set,batch_size=config["batch_size"], shuffle=False,num_workers=config["num_worker"])
        val_loader_s = DataLoader(val_set,batch_size=1, shuffle=False,num_workers=config["num_worker"])
        test_loader_s = DataLoader(test_set,batch_size=1, shuffle=False,num_workers=config["num_worker"])
        model = GlobalLocalBrainAge(config["slice"]*2,patch_size=64,step=32,nblock=6,backbone='vgg8',spatial_size = (H,W))
        model = nn.DataParallel(model).to(device)
        train_validate(model,train_loader,val_loader_s,save_path_s,plane="sagittal",fold=fold,config=config)
        print("---------------------------------------------")
        model=None
        print("Training the model on coronal:")
        torch.manual_seed(seed)
        d = MRI(config["features"],config["excelfile"],config["col_name"],
                slice_number=config["slice"],show_plane = "coronal",discard=discard)  
        a,v = d[0]
        H,W = a.shape[1],a.shape[2]
        save_path_c = config["c_model_dir"]+str(fold)+"/slice"+str(config["slice"])+".pt"
        train_set, val_set,test_set = torch.utils.data.random_split(d, [train_len,val_len,test_len])
        train_loader = DataLoader(train_set,batch_size=config["batch_size"], shuffle=False,num_workers=config["num_worker"])
        val_loader_c = DataLoader(val_set,batch_size=1, shuffle=False,num_workers=config["num_worker"])
        test_loader_c = DataLoader(test_set,batch_size=1, shuffle=False,num_workers=config["num_worker"])
        model = GlobalLocalBrainAge(config["slice"]*2,patch_size=64,step=32,nblock=6,backbone='vgg8',spatial_size = (H,W))
        model = nn.DataParallel(model).to(device)
        train_validate(model,train_loader,val_loader_c,save_path_c,plane="coronal",fold=fold,config=config)
        #perform current fold testing on test data with preloaded test loader from all 3 planes 
        model=None
        test_main(fold,val_loader_a,val_loader_s,val_loader_c,save_path_a,save_path_s,save_path_c,config=config,val=True)
        test_main(fold,test_loader_a,test_loader_s,test_loader_c,save_path_a,save_path_s,save_path_c,config=config,val=False)
