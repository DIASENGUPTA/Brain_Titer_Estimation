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
from datasetravens import get_train_dataloader,get_val_dataloader,get_test_dataloader
import argparse
from sklearn.model_selection import KFold
from itertools import chain
#max_val=0.15
#min_val=0.015



def train_validate(model,data_folder,save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5 )
    scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5,verbose=False)
    criterion = nn.L1Loss()
    best_model_state = None
    best_val_loss=1e+10
    batch_size=64

    #kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    #for batch_idx, batch_data in enumerate(train_dataloader):
    #    print(batch_data)
    #print(train_dataloader)
    #val_dataloader = get_val_dataloader(data_folder, batch_size)
    val_losses=[]
    training_losses=[]
    fig,ax = plt.subplots()
    H,W=130,170
    for fold in range(5):
        train_dataloader = get_train_dataloader(data_folder,fold, batch_size)
        val_dataloader = get_val_dataloader(data_folder,fold, batch_size)
        print('Iterations per epoch',len(train_dataloader))
        #H,W = model.module.H, model.module.W
        for i in range(200):
            running_loss=[]
            model.train()
            print("epoch", i,":")
            maes = torch.zeros((H,W), dtype=torch.float64, device=device)
            counts = torch.zeros((H,W), dtype=torch.int32, device=device)
            for i_batch, (data1,data2, age) in enumerate(train_dataloader):
                #print("Batch No",i_batch)
                #print(data.shape)
                #print(age.shape)
                data1=data1.permute(0,3,1,2)
                data2=data2.permute(0,3,1,2)
                #print(data.shape)
                data1, data2,age = data1.to(device=device,dtype=torch.float),data2.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)  
                
                zlist1 = model(data1)
                zlist2 = model(data2)
                #zlist = torch.cat((zlist1, zlist2), dim=1)
                #print(zlist[0].shape)
                #print(len(zlist))
                zlist1new= torch.stack(zlist1[1:],dim=1)
                zlist2new= torch.stack(zlist2[1:],dim=1)
                zlistfirst = torch.cat((zlist1new, zlist2new), dim=1)
                zlistsecond= torch.cat((zlist1[0], zlist2[0]), dim=1)
                agefirst=age.expand(-1, 24)
                agesecond=age.expand(-1, 2)
                #zlist=torch.mean(zlist, dim=1, keepdim=True)
                #print(zlist.shape)
                #z=torch.mean(zlist)
                #print(zlist1.shape)
                age1 = age.expand(-1, 12)
                #age = torch.permute(age,(1,0))
                #print(age.shape)
                #raise
                #print(age)
                boxes = model.module.location
                #print("Boxes Shape",len(boxes))
                with torch.no_grad():
                    maes,counts = update_heatmap(zlist1new.squeeze(),age1,boxes,maes,counts)
                #b,n,c = zlist.shape
                #print(zlist.shape)
                #age = torch.broadcast_to(age,(n,b))
                #age = torch.permute(age,(1,0))
                # zlist = torch.stack(zlist,dim=1)
                # z=torch.mean(zlist)
                #print('Predicted',zlist)
                #age = age.expand(-1, 2)
                #print(age)
                # Reshape the tensor to shape (64, 13, 1)
                #age = age.unsqueeze(-1)
                #print(age)
                #print(zlist.shape)
                #print(age.shape)
                
                #zlist = torch.squeeze(zlist)
                loss1 = criterion(zlistfirst.squeeze(),agefirst)
                loss2 =criterion(zlistsecond.squeeze(),agesecond)
                #loss3=criterion(zlist2[0],age)
                loss=(loss1+loss2)/2
                running_loss.append(loss.item())
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                #print("Done")
            train_loss=sum(running_loss)/len(running_loss)
            with torch.no_grad():
                #print(maes)
                #maes=maes/counts
                # maes = ( 1-( (maes-torch.amin (maes) ) /( torch.amax(maes)-torch.amin(maes) ) )*255).cpu().numpy().astype(np.uint8)
                # heatmap_img = cv2.applyColorMap(maes, cv2.COLORMAP_JET)
                # #kernel = np.ones((5,5),np.float32)/25
                # #heatmap_img = cv2.filter2D(heatmap_img,-1,kernel)
                # plt.imshow(heatmap_img)
                # plt.savefig("/projectnb/bucbi/Kathakoli/GLT_new"+"/heatmap/validation_heatmap_epoch_"+str(fold)+str(i)+".png")
                # Normalize the maes tensor
                maes_normalized = (maes - torch.min(maes)) / (torch.max(maes) - torch.min(maes))
                maes_normalized = maes_normalized.cpu().numpy()

                # Apply the color map
                heatmap_img = cv2.applyColorMap((maes_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Apply the filter
                kernel = np.ones((5, 5), np.float32) / 25
                heatmap_img = cv2.filter2D(heatmap_img, -1, kernel)
                heatmap_img = heatmap_img.transpose(1, 0, 2) 

                # Display and save the heatmap
                plt.imshow(heatmap_img)
                plt.savefig("/projectnb/bucbi/Kathakoli/GLT_new_ravens" + "/heatmap/validation_heatmap_epoch_" + str(fold) + str(i) + ".png")

            print("Training loss:",train_loss)
            training_losses.append(train_loss)
            avg_val_loss,val_preds,val_targets=[],[],[]
            model.eval()
            print('valloader',len(val_dataloader))
            with torch.no_grad():
                for i_batch, (data1,data2, age) in enumerate(val_dataloader):
                    #a=age[0].item()
                    data1=data1.permute(0,3,1,2)
                    data2=data2.permute(0,3,1,2)
                    data1,data2, age = data1.to(device=device,dtype=torch.float),data2.to(device=device,dtype=torch.float),age.to(device=device,dtype=torch.float)
                    if any([math.isnan(a) for a in age]):
                        continue
                    #print(age)
                    #val_targets=list(chain.from_iterable(age))
                    #ge1+=age.tolist()
                    val_targets = [tensor.item() for tensor in age]
                    #print(len(val_targets))
                    zlist1 = model(data1)
                    zlist2 = model(data2)
                    #zlist = model(data)
                    with torch.no_grad():
                        #zlist1 = torch.stack(zlist[1:],dim=1)
                        zlist1new= torch.stack(zlist1[1:],dim=1)
                        zlist2new= torch.stack(zlist2[1:],dim=1)
                        zlistsecond = torch.cat((zlist1new, zlist2new), dim=1)
                        zlistfirst= torch.cat((zlist1[0], zlist2[0]), dim=1)
                        #b,n,c = zlist.shape
                        zlistpred=torch.cat((torch.stack(zlist1,dim=1),torch.stack(zlist2,dim=1)),dim=1)
                        zlist = torch.squeeze(zlistpred)
                        agesecond=age.expand(-1, 24)
                        agefirst=age.expand(-1, 2)
                        #age1 = age.expand(-1, 12)
                        #age2 = torch.permute(age1,(1,0))
                        #zlist2=torch.stack(zlist,dim=1)
                        
                        p=torch.mean(zlist,dim=1)
                        #print(zlist.shape)
                        #print(p.shape)
                        #print(age.shape)
                        #val_preds+=p.tolist()
                        #age2+=p.tolist()
                        val_preds=[tensor.item() for tensor in p]
                        #val_preds=list(chain.from_iterable(p))
                #         age = age.expand(-1, 13)

                # # Reshape the tensor to shape (64, 13, 1)
                #         age = age.unsqueeze(-1)
                        #age = torch.broadcast_to(age,(n,b))
                        #age = torch.squeeze(torch.permute(age,(1,0)))
                        val_loss1 = criterion(zlistfirst.squeeze(),agefirst)
                        val_loss2 =criterion(zlistsecond.squeeze(),agesecond)
                        val_loss=(val_loss1+val_loss2)/2
                        #val_loss = criterion(zlist,torch.squeeze(age))
                    avg_val_loss.append(val_loss)
                scheduler.step()
                print('Val_targets',len(val_targets))
                print('Val preds',len(val_preds))
                sorted_ind = [i[0] for i in sorted(enumerate(val_targets), key=lambda x:x[1])]
                val_targets.sort()
                val_preds = [val_preds[i] for i in sorted_ind]
                val_loss_curr = sum(avg_val_loss)/len(avg_val_loss)
            plt.cla()
            #print(val_targets)
            #print(val_preds)
            ax.scatter(val_targets,val_preds)
            #plt.plot([min_val,max_val],[min_val,max_val],'k-',lw=2)
            cor=np.corrcoef(val_targets,val_preds)[0,1]
            plt.suptitle("pearson correlation:"+str(cor))
            plt.title("validation mae loss:"+str(val_loss_curr.item()))
            plt.savefig("/projectnb/bucbi/Kathakoli/GLT_new_ravens"+"/plots/validation_epoch_"+str(fold)+str(i)+".png")
            plt.cla()
            print("validation mae loss:", val_loss_curr.item())
            val_losses.append(val_loss_curr.item())

            if val_loss_curr<best_val_loss:
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), str(fold)+save_path)
                best_val_loss = val_loss_curr
        plt.figure(figsize=(12,6))
        plt.xlabel("epoch")
        plt.ylabel("validation loss")
        plt.plot(val_losses,label='validation_losses')
        plt.plot(training_losses,label='training_losses')
        #ax.set_xlim(0,60)
        #ax.set_ylim(0,0.3)
        plt.autoscale()
        plt.legend()
        plt.savefig("/projectnb/bucbi/Kathakoli/GLT_new_ravens"+"/plots/val_loss_epoch"+str(fold)+".png")
        plt.cla()
        plt.clf()
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), save_path)
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 
    parser = argparse.ArgumentParser(description='bucbi')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    model = GlobalLocalBrainAge(5,patch_size=64,step=32,nblock=6,backbone='vgg8')
    model = nn.DataParallel(model).to(device)
    train_validate(model,args.data_folder,args.save_path)