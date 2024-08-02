import time
import random
import argparse

import torch
#import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from FiANet import fusNet
from dataset_front import get_dataloader



def train(data_folder, save_path):
    """
    Function for training the network. You can make changes (e.g., add validation dataloader, change batch_size and #of epoch) accordingly.
    """
    
    device = torch.device('cuda')
    nr_epochs = 30
    batch_size = 1
    #nr_of_classes = 7  # needs to be changed
    start_time = time.time()

    infer_action = fusNet()
    infer_action1=nn.DataParallel(infer_action)
    infer_action1.to(device)
    criterion=nn.L1Loss()#used Mean-squared error loss as the loss
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=0.001)
    #print('Hi')

    train_loader = get_dataloader(data_folder, batch_size)
    #print(train_loader.shape)
    loss_values=[]
    for epoch in range(nr_epochs):
        total_loss = 0
        print("hello")
        batch_in1 = []
        batch_in2=[]
        batch_gt = []
        #print("Hi")

        for batch_idx, batch in enumerate(train_loader):
            batch_in1, batch_in2, batch_gt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            print(batch_gt)
            #print('Hey')
            #x = torch.rand(1,1,64,64,64).to(device)
            #y = torch.rand(1,1,64,64,64).to(device)
            batch_out = infer_action1(batch_in1.float().unsqueeze(0), batch_in2.float().unsqueeze(0))
            #print(batch_out)
            #batch_out=infer_action(x,y)
            #batch_out1=infer_action.b(batch_out)
            loss1 = criterion(batch_out[0], batch_gt)
            loss2= criterion(batch_out[1], batch_gt)
            loss3=criterion(batch_out[2], batch_gt)
            loss=loss1+loss2+loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #average_loss=total_loss/(batch_idx+1)
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))
        loss_values.append(total_loss)
    torch.save(infer_action, save_path)
    plt.plot(loss_values)
    plt.savefig('/projectnb/bucbi/Kathakoli/loss_front.png')


#def cross_entropy_loss(batch_out, batch_gt):
    """
    Calculates the cross entropy loss between the prediction of the network and
    the ground truth class for one batch.
                    C = number of classes
    batch_out:      torch.Tensor of size (batch_size, C)
    batch_gt:       torch.Tensor of size (batch_size, C)
    return          float
    """
    # compute the log-softmax of the output predictions
    #log_probs = batch_out - torch.logsumexp(batch_out, dim=1, keepdim=True)
    
    # get the log-probability of the ground truth class for each example in the batch
    #batch_gt_log_probs = torch.sum(batch_gt * log_probs, dim=1)
    
    # compute the negative average log-likelihood over the batch
    #loss = -torch.mean(batch_gt_log_probs)
    
    #return loss
    #loss=nn.functional.binary_cross_entropy_with_logits(batch_out,batch_gt)
    #return loss

'''
def mse_loss(batch_out,batch_gt):
    loss=nn.MSELoss(batch_out,batch_gt)
    return loss
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bucbi')
    parser.add_argument('-d', '--data_folder', default="./", type=str, help='path to where you save the dataset you collect')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in .pth format')
    args = parser.parse_args()
    
    train(args.data_folder, args.save_path)