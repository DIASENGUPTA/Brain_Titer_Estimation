import numpy as np
import cv2
import torch 
import torch.nn.functional as F


def update_heatmap(predictions,target,boxes,maes,counts):
    #H,W = HW[0],HW[1]
    b = predictions.shape[0]
    if predictions.shape[0]<64:
        predictions=F.pad(predictions, (0, 0, 0, 64-predictions.shape[0]), value=0)
        target=F.pad(target, (0, 0, 0, 64-target.shape[0]), value=0)
    err = torch.sum(torch.abs(target-predictions),dim=0) #64,15
    for box,pred in zip(boxes,err):
        y_start,y_end,x_start,x_end = box[0],box[1],box[2],box[3]
        #print(len(pred.unsqueeze(-1)))
        #print(b)
        maes[y_start:y_end,x_start:x_end] += pred.unsqueeze(-1)
        counts[y_start:y_end,x_start:x_end]+=b
       
        
    return maes,counts



# def update_heatmap(predictions, target, boxes, maes, counts):
#     b = predictions.shape[0]
#     err = torch.sum(torch.abs(target - predictions), dim=0)  # 64, 15
#     for box, pred in zip(boxes, err):
#         y_start, y_end, x_start, x_end = box[0], box[1], box[2], box[3]
#         try:
#             maes[x_start:x_end, y_start:y_end] += pred.unsqueeze(-1)  # Swap x and y indices
#             counts[x_start:x_end, y_start:y_end] += b  # Swap x and y indices
#         except RuntimeError as e:
#             print("Error: Dimension mismatch between maes slice and pred tensor")
#             continue  # Handle the situation by skipping this iteration or modifying pred

#     return maes, counts