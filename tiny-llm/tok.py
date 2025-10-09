import torch 

# most simple tokenizer just convert a string to its byte 

class ByteTokenizer:
    def encode(self, s:str):
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)
    
    def decode(self, s):
        if isinstance(s, torch.Tensor):
            s = s.tolist()
        return bytes(s).decode('utf-8',errors='ignore')
    
    @property
    def vocab_size(self):
        return 256