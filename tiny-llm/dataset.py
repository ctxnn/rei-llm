import torch 
import pathlib 

class ByteDataset:
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        self.block_size = block_size 
        data = pathlib.Path(path).read_bytes() # reads the entire dataset as raw bytes
        data = torch.tensor(list(data), dtype = torch.long)
        n = int(len(data)*split)
        self.train = data[:n]
        self.val = data[n:]
        
    def get_batch(self, which: str, batch_size: int, device: torch.device):
        buf = self.train if which == 'train' else self.val # the entire text file as bytes
        assert len(buf) > self.block_size + 1, 'file too small for given block_size'
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,)) # gives random windows to go through 
        x = torch.stack([buf[i:i+self.block_size] for i in ix])
        y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix]) # because its a autoregressive model 
        return x.to(device), y.to(device)
        