import torch 
import torch.nn as nn 
import math 


# .register in python means that THESE ARE NOT LEARNABLE PARAMETERS they need to be stored as it is, they are absolute
# we are dealing with absolute

    
class PoE(nn.Module):
    def __init__(self, max_len:int, d_model:int):
        super().__init__()
        self.max_len = max_len 
        self.d_model = d_model 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # formula from 2017 paper
        divterm = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * divterm)
        pe[:,1::2] = torch.cos(position * divterm)
        self.register_buffer('pe',pe) # (max_len, d_model)
        
    def forward(self, x:torch.Tensor):
        B, T, _ = x.shape   
        return x + self.pe[:T].unsqueeze(0)
        
        
        
        
    