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
from heatmap import update_heatmap
from dataset3 import get_train_dataloader,get_val_dataloader,get_test_dataloader
import argparse
from itertools import chain

def test(model,val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_preds=[]
    val_targets=[]
    model.eval()
    with torch.no_grad():
        for i_batch, datas in enumerate(val_dataloader):
           
            data1,data2,age=datas
            data1,data2, age = data1.to(device=device,dtype=torch.float),data2.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
            if any([math.isnan(a) for a in age]):
                continue
            data1=data1.permute(0,3,1,2)
            data2=data2.permute(0,3,1,2)
            val_targets+=age.tolist()
            zlist1 = model(data1)
            zlist2 = model(data2)
            # for z in zlist:
            #     val_preds+=z.to_list()[0]
            zlist1 = torch.stack(zlist1,dim=1)
            zlist2 = torch.stack(zlist2,dim=1)
            zlist = torch.cat((zlist1, zlist2), dim=1)
            zlist = torch.squeeze(zlist)
            #print(zlist.shape)
            p=torch.mean(zlist,dim=1)
            val_preds+=p.tolist()
            print(len(val_preds))
            print(len(val_targets))
            flattened_list = list(chain(*val_targets))
            print(flattened_list)
    return val_preds,flattened_list

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 
    axial_model = GlobalLocalBrainAge(5,patch_size=64,step=32,nblock=6,backbone='vgg8')
    axial_model = nn.DataParallel(axial_model).to(device)
    axial_model.load_state_dict(torch.load("/projectnb/bucbi/Kathakoli/GLT/ravens_train.pt"))
    test_dataloader=get_test_dataloader("/projectnb/bucbi", 1,64)
    predictions,targets = test(axial_model,test_dataloader)
    targets,sorted_ind = torch.sort(torch.FloatTensor(targets))
    print("Max and min:",torch.max(targets), torch.min(targets))
    criterion = nn.L1Loss()
    predictions = np.array([predictions[i] for i in sorted_ind])
    #pred_avg = np.squeeze(np.average(predictions.reshape((-1,178)),axis=1))
    #target_avg = np.squeeze(np.average(targets.reshape((-1,178)),axis=1))
    #print(predictions)
    #print(targets)
    #mae = criterion(torch.tensor(pred_avg),torch.tensor(target_avg))
    
    val=False
    mae = criterion(torch.FloatTensor(predictions),targets)
    if val==True:
        print("MAE on validation set with predictions averaging on 3 models:", mae)
    else:
        print("MAE on test set with predictions averaging on 3 models:", mae)
    fig,ax = plt.subplots()
    ax.scatter(targets,predictions)
    cor=np.corrcoef(targets,predictions)[0,1]
    plt.suptitle("correlation:"+str(cor))
    if val==True:
        plt.title("validation mae on fold"+":"+str(mae))
        plt.savefig("val"+".png")
    else:
        plt.title("test mae on fold"+":"+str(mae))
        plt.savefig("GLT_ravens_final"+".png")