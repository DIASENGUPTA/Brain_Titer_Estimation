import time
import random
import argparse

import torch
#import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#from Fia import fusNet
from dataset import get_dataloader
import glob

import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import nibabel as nib
import torchio as tio
import skimage.transform as skTrans
import cv2

device = torch.device('cuda')
data_dir = '/projectnb/bucbi/'
subjectInfoPath = '/projectnb/bucbi/Kathakoli/BrainageMeas_Kathakoli.xlsx'
subjectInfo = pd.read_excel(subjectInfoPath)
subjectInfo = subjectInfo[['participant','Age']]
s1=np.array(subjectInfo['participant'])
s2=np.array(subjectInfo['Age'])
newlist=[]
        #self.data_list1 = glob.glob(data_dir+'NACC_T1/*/*_MNI1mmbrainfinal.nii.gz') #need to change to your data format
        #self.data_list2 = glob.glob(data_dir+'NACC_T1/*/ravens_in_MNI1mm.nii.gz')
        #self.data_list3 = glob.glob(data_dir+'/*/*_MNI1mmbrainfinal.nii.gz')
data_list1=[]
data_list2=[]
agelist=[]
for i1 in ['NACC_T1','UMass_T1','HCP_T1','ADNI_ML_images/ADNI1_CN','ADNI_ML_images/ADNI2_CN']:
        if i1=='NACC_T1':
                with open('/projectnb/bucbi/NACC_T1/input_list_NACC.txt') as file:
                    for line in file:
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        data_list1.append(input_file1)
                        data_list2.append(input_file2)
                        for i in range(len(s1)):
                            if s1[i]==input_folder:
                                agelist.append(s2[i])
        elif i1=='UMass_T1':
                with open('/projectnb/bucbi/UMass_T1/input_list_UMass.txt') as file:
                    for line in file:
                        temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/*{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrain*.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        data_list1.append(input_file1)
                        data_list2.append(input_file2)
                        check= (input_file1.split('/')[-2]).split('_')[-1]
                        for i in range(len(s1)):
                            #print("hi")
                            #print(self.s1[i])
                            #print(check)
                            if str(s1[i]).split('_')[-1]==check:
                                #print("hello")
                                temp.append(s2[i])
                        #print(self.temp)
                        #print(self.temp)
                        if(len(temp)>1):
                            if(temp[1])=='nan':
                                temp[1]=60
                            agelist.append(temp[1])
                        else:
                            if(temp[0])=='nan':
                                temp[0]=60
                            agelist.append(temp[0])
        elif i1=='HCP_T1':
                with open('/projectnb/bucbi/HCP_T1/input_list_HCP.txt') as file:
                    for line in file:
                        temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/brain_MNI1mm.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        data_list1.append(input_file1)
                        data_list2.append(input_file2)
                        for i in range(len(s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(s1[i])==input_folder:
                                #print("hello")
                                temp.append(s2[i])
                        #print(self.temp)
                        if(len(temp)>1):
                            if(temp[1])=='nan':
                                temp[1]=60
                            agelist.append(temp[1])
                        else:
                            if(temp[0])=='nan':
                                temp[0]=60
                            agelist.append(temp[0])
        elif i1=='ADNI_ML_images/ADNI1_CN':
                with open('/projectnb/bucbi/ADNI_ML_images/ADNI1_CN/input_list_adni1.txt') as file:
                    for line in file:
                        temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mm*.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        data_list1.append(input_file1)
                        data_list2.append(input_file2)
                        for i in range(len(s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(s1[i])==input_folder:
                                #print("hello")
                                temp.append(s2[i])
                        #print(self.temp)
                        if(temp[0])=='nan':
                            temp[0]=60
                        agelist.append(temp[0])
        elif i1=='ADNI_ML_images/ADNI2_CN':
                with open('/projectnb/bucbi/ADNI_ML_images/ADNI2_CN/input_list_adni2.txt') as file:
                    for line in file:
                        temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mm*.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        data_list1.append(input_file1)
                        data_list2.append(input_file2)
                        for i in range(len(s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(s1[i])==input_folder:
                                #print("hello")
                                temp.append(s2[i])
                        #print(self.temp)
                        if(temp[0])=='nan':
                            temp[0]=60
                        #self.newlist.append(self.temp[0])
                        agelist.append(temp[0])
transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
transform2= tio.CropOrPad((64,64,64))
model = torch.load('/projectnb/bucbi/Kathakoli/modelfusonly.pth')
output_list=[]
#k=0
with torch.no_grad():
    for i in range(len(data_list1)):
        img1 = nib.load(data_list1[i]).get_fdata()
        img2= nib.load(data_list2[i]).get_fdata()
        imgnew1=skTrans.resize(img1, (64,64,64), order=1, preserve_range=True)
        imgnew2=skTrans.resize(img2, (64,64,64), order=1, preserve_range=True)
        image1 = transform(imgnew1)#numpy is converted to tensor
        image2= transform(imgnew2) 
        age=torch.as_tensor([agelist[i]])
        #model.regression_3d.weight=nn.Parameter(model.regression_3d.weight.transpose(0, 1))
        #print(model.regression_3d.weight.shape)
        print(image1.float().unsqueeze(0).shape)
        output1, output2, output3 = model.forward(image1.float().unsqueeze(0).unsqueeze(0).to(device),image2.float().unsqueeze(0).unsqueeze(0).to(device))
        print(output1)
        print(output2)
        print(output3)
        output_list.append(output3.detach().cpu().numpy()[0][0])
        #k+=1
        #break

plt.figure()
#print(agelist[:1])
#print(output_list)
plt.plot(agelist, 'go', output_list, 'r*')
#plt.plot(output_list)
plt.savefig('/projectnb/bucbi/Kathakoli/corr_CN_CN_fus_new.png')
#plt.show()
