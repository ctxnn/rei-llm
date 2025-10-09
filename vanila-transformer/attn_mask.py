import torch 
import torch.nn as nn 

# here we are basically making the model not see the future tokens by masking the attention matrix ( upper triangular matrix)
# softmax(-inf) = 0
def casual_mask(T:int, device = None):
    m = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, 1, T, T)
    