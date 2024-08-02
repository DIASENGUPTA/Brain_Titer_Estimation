import torch
import torch.nn as nn
import copy
import math
#import vgg as vnet
#import incep as i
import Novel as ni

class GlobalAttention(nn.Module):
    def __init__(self, 
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()
        
        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self,locx,glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)
        
        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        return attention_output

class convBlock(nn.Module):
    def __init__(self,inplace,outplace,kernel_size=3,padding=1):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace,outplace,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x
    
class Feedforward(nn.Module):
    def __init__(self,inplace,outplace):
        super().__init__()
        
        self.conv1 = convBlock(inplace,outplace,kernel_size=1,padding=0)
        self.conv2 = convBlock(outplace,outplace,kernel_size=1,padding=0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class GlobalLocalBrainAge(nn.Module):
    def __init__(self,inplace,
                 patch_size=64,
                 step=-1,
                 nblock=6,
                 drop_rate=0.5,
                 backbone='Novel'):
        """
        Parameter:
            @patch_size: the patch size of the local pathway
            @step: the step size of the sliding window of the local patches
            @nblock: the number of blocks for the Global-Local Transformer
            @Drop_rate: dropout rate
            @backbone: the backbone of extract the features
        """
        
        super().__init__()
        
        self.patch_size = patch_size
        self.step = step
        self.nblock = nblock
        if self.step <= 0:
            self.step = int(patch_size//2)
            
        
        self.global_feat = ni.Novel(inplace)
        self.local_feat = ni.Novel(inplace)
        hidden_size = 512
        

        #print(self.global_feat)
    
        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()
        
        for n in range(nblock):
            atten = GlobalAttention(
                    transformer_num_heads=8,
                    hidden_size=hidden_size,
                    transformer_dropout_rate=drop_rate)
            self.attnlist.append(atten)
            
            fft = Feedforward(inplace=hidden_size*2,
                              outplace=hidden_size)
            self.fftlist.append(fft)
            
        self.avg = nn.AdaptiveAvgPool2d(1)
        out_hidden_size = hidden_size
            
        self.gloout = nn.Linear(out_hidden_size,1)
        self.locout = nn.Linear(out_hidden_size,1)
        self.location=[]
        self.done=False
    def forward(self,xinput):
        _,_,H,W=xinput.size()
        #print(_,_,H,W)
        outlist = []
        #inference on backbone for feature encoding
        xglo = self.global_feat(xinput)
        #print(xglo.shape)
        xgfeat = torch.flatten(self.avg(xglo),1)
        #print(xgfeat.shape)
        #feature embedding to desired dimension
        glo = self.gloout(xgfeat)
        outlist=[glo]
        
        B2,C2,H2,W2 = xglo.size()
        xglot = xglo.view(B2,C2,H2*W2)
        xglot = xglot.permute(0,2,1)
        #xglot = globale feature with shape:(batch, flatten_spatial, channel)
        for y in range(0,H-self.patch_size+1,self.step):
            for x in range(0,W-self.patch_size+1,self.step):
                
                if not self.done:
                    self.location.append([y,y+self.patch_size,x,x+self.patch_size])
                locx = xinput[:,:,y:y+self.patch_size,x:x+self.patch_size]
                #print(locx.shape)
                xloc = self.local_feat(locx)
                #print(xloc.shape)
                
                for n in range(self.nblock):
                    B1,C1,H1,W1 = xloc.size()
                    #print("LOC dim")
                    #print(B1,C1,H1,W1)
                    xloct = xloc.view(B1,C1,H1*W1)
                    xloct = xloct.permute(0,2,1)
                    #xloct = local feature with shape:(batch,flatten_spatial, channel)
                    tmp = self.attnlist[n](xloct,xglot)
                    #tmp = global context infused local feature with attention 
                    tmp = tmp.permute(0,2,1)
                    tmp = tmp.view(B1,C1,H1,W1)
                    
                    tmp = torch.cat([tmp,xloc],1)
                    #tmp = channel-wise combined global-local feature and current xloc
                    tmp = self.fftlist[n](tmp)
                    xloc = xloc + tmp
                    
                xloc = torch.flatten(self.avg(xloc),1)
                    
                out = self.locout(xloc)
                outlist.append(out)
        self.done=True 
        #print(len(outlist))
        return outlist

        