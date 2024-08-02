import os
import torch
import torch.nn as nn
import nibabel as nib
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
from GlobalLocalTransformer_Novel import GlobalAttention,convBlock,GlobalLocalBrainAge
import pandas as pd
import math
import random
import json
import pickle
import cv2
#from heatmap import update_heatmap

max_val=0.15
min_val=0.015

def test(model,val_dataloader):
    device = torch.device("cpu")
    val_preds=[]
    val_targets=[]
    model.eval()
    with torch.no_grad():
        for i_batch, datas in enumerate(val_dataloader):
           
            data,age=datas
            data, age = data.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
            if any([math.isnan(a) for a in age]):
                continue
            val_targets+=age.tolist()
            zlist = model(data)
            zlist = torch.stack(zlist,dim=1)
            zlist = torch.squeeze(zlist)
            p=torch.mean(zlist,dim=1)
            val_preds+=p.tolist()
    return val_preds,torch.FloatTensor(val_targets)

def train_validate(model,train_dataloader,val_dataloader,save_path,plane="axial",fold=0,config=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"] )
    scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5,verbose=False)
    criterion = nn.L1Loss()
    boxes = model.module.location
    best_val_loss=1e+10
    val_losses=[]
    training_losses=[]
    fig,ax = plt.subplots()
    H,W = model.module.HW[0], model.module.HW[1]
    for i in range(config["epoch"]):
        running_loss=[]
        model.train()
        print("epoch", i,":")
        maes = torch.zeros((H,W), dtype=torch.float64, device=device)
        counts = torch.zeros((H,W), dtype=torch.int32, device=device)
        for i_batch, (data, age) in enumerate(train_dataloader):
            data, age = data.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
            #print(data.shape)  
            zlist = model(data)
            #print(zlist.shape)
            #zlist = torch.stack(zlist,dim=1)
            zlist = torch.stack(zlist,dim=1)
            #print(zlist.shape)
            b,n,c = zlist.shape
            #with torch.no_grad():
            #    maes,counts = update_heatmap(torch.squeeze(zlist[:,1:,:]),torch.unsqueeze(age,dim=1),boxes,(H,W),maes,counts)
            zlist = torch.squeeze(zlist)
            #print(age.shape)
            age = torch.broadcast_to(age,(n,b))
            #print(age.shape)
            age = torch.permute(age,(1,0))
            loss = criterion(zlist,age)
            running_loss.append(loss.item())
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss=sum(running_loss)/len(running_loss)
        with torch.no_grad():
            maes=maes/counts
            maes = ( 1-( (maes-torch.amin (maes) ) /( torch.amax(maes)-torch.amin(maes) ) )*255).cpu().numpy().astype(np.uint8)
            heatmap_img = cv2.applyColorMap(maes, cv2.COLORMAP_JET)
            kernel = np.ones((5,5),np.float32)/25
            heatmap_img = cv2.filter2D(heatmap_img,-1,kernel)
            plt.imshow(heatmap_img)
            plt.savefig(config["root_dir"]+plane+"/fold"+str(fold)+"/heatmap/validation_heatmap_epoch"+str(i)+".png")
        print("Training loss:",train_loss)
        training_losses.append(train_loss)
        avg_val_loss,val_preds,val_targets=[],[],[]
        model.eval()
        with torch.no_grad():
            for i_batch, (data, age) in enumerate(val_dataloader):
                a=age[0].item()
                data, age = data.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
                if any([math.isnan(a) for a in age]):
                    continue
                val_targets+=age.tolist()
                zlist = model(data)
                with torch.no_grad():
                    zlist = torch.stack(zlist,dim=1)
                    b,n,c = zlist.shape
                    zlist = torch.squeeze(zlist)
                    p=torch.mean(zlist,dim=1)
                    val_preds+=p.tolist()
                    age = torch.broadcast_to(age,(n,b))
                    #print(age.shape)
                    age = torch.squeeze(torch.permute(age,(1,0)))
                    val_loss = criterion(zlist,age)
                avg_val_loss.append(val_loss)
            scheduler.step()
            sorted_ind = [i[0] for i in sorted(enumerate(val_targets), key=lambda x:x[1])]
            val_targets.sort()
            val_preds = [val_preds[i] for i in sorted_ind]
            val_loss_curr = sum(avg_val_loss)/len(avg_val_loss)
        plt.cla()
        ax.scatter(val_targets,val_preds,vmin=min_val, vmax=max_val)
        plt.plot([min_val,max_val],[min_val,max_val],'k-',lw=2)
        cor=np.corrcoef(val_targets,val_preds)[0,1]
        plt.suptitle("pearson correlation:"+str(cor))
        plt.xlabel("target "+config["col_name"])
        plt.ylabel("predicted "+config["col_name"])
        plt.title("validation mae loss:"+str(val_loss_curr.item()))
        plt.savefig(config["root_dir"]+plane+"/fold"+str(fold)+"/plots/validation_epoch"+str(i)+".png")
        plt.cla()
        print("validation mae loss:", val_loss_curr.item())
        val_losses.append(val_loss_curr.item())

        if val_loss_curr<best_val_loss:
            torch.save(model.state_dict(), save_path)
            best_val_loss = val_loss_curr
    plt.figure(figsize=(12,6))
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.plot(val_losses,label='validation_losses')
    plt.plot(training_losses,label='training_losses')
    ax.set_xlim(0,60)
    ax.set_ylim(0,0.3)
    plt.autoscale()
    plt.legend()
    #plt.savefig(config["root_dir"]+plane+"/fold"+str(fold)+"/plots/val_loss_epoch.png")
    plt.cla()
    plt.clf()
