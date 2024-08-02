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


class BrainAgeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.subjectInfoPath = '/projectnb/bucbi/Kathakoli/BrainageMeas_Kathakoli.xlsx'
        self.subjectInfo = pd.read_excel(self.subjectInfoPath)
        self.subjectInfo = self.subjectInfo[['participant','Age']]
        self.s1=np.array(self.subjectInfo['participant'])
        self.s2=np.array(self.subjectInfo['Age'])
        self.newlist=[]
        #self.data_list1 = glob.glob(data_dir+'NACC_T1/*/*_MNI1mmbrainfinal.nii.gz') #need to change to your data format
        #self.data_list2 = glob.glob(data_dir+'NACC_T1/*/ravens_in_MNI1mm.nii.gz')
        #self.data_list3 = glob.glob(data_dir+'/*/*_MNI1mmbrainfinal.nii.gz')
        self.data_list1=[]
        self.data_list2=[]
        self.agelist=[]
        for i1 in ['Gulf_war_T1_T2','GW_session2','GW_Alabama','GW_UCSF','GW_Georgetown']:
        #for i1 in ['Gulf_war_T1_T2','UMass_T1','HCP_T1','NACC_T1','ADNI_ML_images/ADNI1_CN','ADNI_ML_images/ADNI2_CN','GW_session2','GW_Alabama','GW_UCSF','GW_Georgetown']:
            if i1=='NACC_T1':
                with open('/projectnb/bucbi/NACC_T1/input_list_NACC.txt') as file:
                    for line in file:
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            if self.s1[i]==input_folder:
                                self.agelist.append(self.s2[i])
            elif i1=='Gulf_war_T1_T2':
                with open('/projectnb/bucbi/Gulf_war_T1_T2/input_list_gwt1t2.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(self.s1[i])==input_folder:
                                #print("hello")
                                self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(self.temp[0])=='nan':
                            self.temp[0]=60
                        self.agelist.append(self.temp[0])
            elif i1=='GW_session2':
                with open('/projectnb/bucbi/GW_session2/input_list_GW2.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(self.s1[i])==input_folder:
                                #print("hello")
                                self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(len(self.temp)>1):
                            if(self.temp[1])=='nan':
                                self.temp[1]=60
                            self.agelist.append(self.temp[1])
                        else:
                            if(self.temp[0])=='nan':
                                self.temp[0]=60
                            self.agelist.append(self.temp[0])
            #elif i1=='UMass_T1':
            #    with open('/projectnb/bucbi/UMass_T1/UMass_new.txt') as file:
            #        for line in file:
            #            self.temp=[]
            #            input_folder = line.strip()
            #            input_f_name = f"{data_dir}/{i1}/*{input_folder}"
            #            #print(input_f_name)
            #            input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
            #            input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
            #            self.data_list1.append(input_file1)
            #            self.data_list2.append(input_file2)
            #            check= input_file1.split('/')[-2].split('/')[-1]
            #            for i in range(len(self.s1)):
            #                #print("hi")
            #                #print(type(self.s1[0]))
            #                #print(type(input_folder))
            #                if str(self.s1[i])==check:
            #                    #print("hello")
            #                    self.temp.append(self.s2[i])
            #            #print(self.temp)
            #            print(self.temp)
            #            if(len(self.temp)>1):
            #                if(self.temp[1])=='nan':
            #                    self.temp[1]=60
            #                self.agelist.append(self.temp[1])
            #            else:
            #                if(self.temp[0])=='nan':
            #                    self.temp[0]=60
            #                self.agelist.append(self.temp[0])
            elif i1=='HCP_T1':
                with open('/projectnb/bucbi/HCP_T1/input_list_HCP.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/brain_MNI1mm.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(self.s1[i])==input_folder:
                                #print("hello")
                                self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(len(self.temp)>1):
                            if(self.temp[1])=='nan':
                                self.temp[1]=60
                            self.agelist.append(self.temp[1])
                        else:
                            if(self.temp[0])=='nan':
                                self.temp[0]=60
                            self.agelist.append(self.temp[0])
            elif i1=='ADNI_ML_images/ADNI1_CN':
                with open('/projectnb/bucbi/ADNI_ML_images/ADNI1_CN/input_list_adni1.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mm*.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(self.s1[i])==input_folder:
                                #print("hello")
                                self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(self.temp[0])=='nan':
                            self.temp[0]=60
                        self.agelist.append(self.temp[0])
            elif i1=='ADNI_ML_images/ADNI2_CN':
                with open('/projectnb/bucbi/ADNI_ML_images/ADNI2_CN/input_list_adni2.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mm*.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            if str(self.s1[i])==input_folder:
                                #print("hello")
                                self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(self.temp[0])=='nan':
                            self.temp[0]=60
                        #self.newlist.append(self.temp[0])
                        self.agelist.append(self.temp[0])
            elif i1=='GW_Alabama':
                with open('/projectnb/bucbi/GW_Alabama/input_list_albama.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            #print(input_folder.split('_')[-2][:-1]+'t_'+input_folder.split('_')[-1])
                            if str(self.s1[i])==input_folder.split('_')[-2][:-1]+'t_'+input_folder.split('_')[-1]:
                                #print("hello")
                                self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(self.temp[0])=='nan':
                            self.temp[0]=60
                        #self.newlist.append(self.temp[0])
                        self.agelist.append(self.temp[0])
            elif i1=='GW_UCSF':
                with open('/projectnb/bucbi/GW_UCSF/input_list_ucsf.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            #print(input_folder.split('_')[-2][:-1]+'t_'+input_folder.split('_')[-1])
                            if str(self.s1[i])==input_folder:
                                #print("hello")
                                self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(self.temp[0])=='nan':
                            self.temp[0]=60
                        #self.newlist.append(self.temp[0])
                        self.agelist.append(self.temp[0])
            elif i1=='GW_Georgetown':
                with open('/projectnb/bucbi/GW_Georgetown/input_list_g.txt') as file:
                    for line in file:
                        self.temp=[]
                        input_folder = line.strip()
                        input_f_name = f"{data_dir}/{i1}/*{input_folder}"
                        #print(input_f_name)
                        input_file1= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
                        input_file2= glob.glob(f"{input_f_name}/ravens_in_MNI1mm.nii.gz")[0]
                        self.data_list1.append(input_file1)
                        self.data_list2.append(input_file2)
                        for i in range(len(self.s1)):
                            #print("hi")
                            #print(type(self.s1[0]))
                            #print(type(input_folder))
                            #print(str(self.s1[i]).split('_'))
                            #print(input_folder)
                            if(len(str(self.s1[i]).split('-'))>1):
                                #print(str(self.s1[i]).split('-')[-2]+'-'+str(self.s1[i]).split('-')[-1])
                                if str(self.s1[i]).split('-')[-2]+'-'+str(self.s1[i]).split('-')[-1]==input_folder:
                                #print("hello")
                                    self.temp.append(self.s2[i])
                        #print(self.temp)
                        if(self.temp[0])=='nan':
                            self.temp[0]=60
                        #self.newlist.append(self.temp[0])
                        self.agelist.append(self.temp[0])
            #else:
            #    continue
        
                        
        #print(self.agelist)
        #print(len(self.data_list1))
        #print(len(self.data_list2))
        #print(len(self.agelist))
        #self.s1=np.array(self.subjectInfo['participant'])[2353:2441]
        #self.name=[]
        #[2353:2441]
        #self.s2=np.array(self.subjectInfo['Age'])[2353:2441]
        #print(self.s1[0])
        #print(self.s1[-1])
        #print(self.data_list1)
        #for i in self.data_list1:
        #    self.name.append(i.split('/')[-2].split('/')[-1])
        #self.name=self.data_list1[0].split('/')[-2].split('/')[-1]
        #print(self.name,'name')
        #self.s=0

        #print(self.data_list)

        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        self.transform2= tio.CropOrPad((64,64,64))

    def __len__(self):
        return len(self.data_list1)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        #data = np.load(self.data_list[idx], allow_pickle=True)
        #print("Heyhey")
        img1 = nib.load(self.data_list1[idx]).get_fdata()
        #print(img1.shape)
        img2= nib.load(self.data_list2[idx]).get_fdata()
        #print(img2.shape)
        imgnew1=skTrans.resize(img1, (64,64,64), order=1, preserve_range=True)
        imgnew2=skTrans.resize(img2, (64,64,64), order=1, preserve_range=True)
        #cv2.imshow('BrainImage',imgnew1)
        image1 = self.transform(imgnew1)#numpy is converted to tensor
        image2= self.transform(imgnew2)#resize the image
        
        
        #print(imgnew1.shape)
        #print(imgnew2.shape)
        #print(self.name)
        #for j in range(len(self.data_list1)):
        #    for i in range(2681):
        #        print(j,i)
        #        if(self.name[j]==self.subjectInfo['participant'][i]):
        #            print(self.subjectInfo['participant'][i],'Hi')
        #            self.s=self.subjectInfo['Age'][i]
        #print(type(self.s))
       # age = torch.tensor(self.s,dtype=torch.float64)#convert actions to tensor
        age=torch.as_tensor([self.agelist[idx]])
        #
        #print(image1.shape)
        #print(image2.shape)
        #print(age.shape)

        return (image1, image2, age)


def get_dataloader(data_dir, batch_size, num_workers=4):
    return torch.utils.data.DataLoader(
                BrainAgeDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                #shuffle=shuffle
            )
    