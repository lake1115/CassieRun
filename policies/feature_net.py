#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   feature_net.py
@Time    :   2023/03/16 11:35:37
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   feature_net replace Actor Net
'''

import torch
from linformer import Linformer
from torch import nn
import random
class Feature_net(nn.Module):
    def __init__(self):
        super().__init__()

        self.embeding_layer = nn.Linear(43, 256)
        self.model = Linformer(
            dim = 256,
            seq_len = 28,
            depth = 3,
            heads = 4,
            k = 128,
        )
        self.output_layer = nn.Linear(256,10)
 
    def forward(self, s):
        idx = s[:,:,-1:]
        ss = torch.zeros_like(s)
        indices = torch.argsort(idx, dim=1).squeeze(-1)
        for i in range(len(s)):
            ss[i] = s[i][indices[i]]
        
        #f = self.embeding_layer(s)
        f = self.model(ss[:,:,:2])
        a = self.output_layer(f)
        return a




if __name__ == '__main__':


    # x = torch.randint(1,10000,(1,512))
    # y = torch.randint(1,10000,(1,512))

    # x_mask = torch.ones_like(x).bool()
    # y_mask = torch.ones_like(y).bool()

    # enc_output = encoder(x)
    # print(enc_output.shape)
    # dec_output = decoder(y, embeddings=enc_output)
    # print(dec_output.shape) # (1, 512, 10000)
    model = Feature_net()
    
    x = torch.randn(2,10,2)
    
    idx1 = [i for i in range(10)]
    random.shuffle(idx1)
    idx1= torch.tensor(idx1).unsqueeze(-1).unsqueeze(0)
    idx2 = [i for i in range(10)]
    random.shuffle(idx2)
    idx2= torch.tensor(idx2).unsqueeze(-1).unsqueeze(0)
    idx = torch.cat([idx1,idx2],dim=0)
    xx = torch.cat([x,idx],dim=-1)
    aa = model(xx)
    print(aa)

