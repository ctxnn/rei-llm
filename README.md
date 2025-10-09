# rei-llm 

a large language model, trained from scratch 

from PRE TRAINING -> RLHF 

> [!NOTE]
> i will make this project in parts so all the parts and updates will be in git commit history

### step 1 - *vanilla transformer*

implemented to revise and see how the transformer works

1. `attn_mask.py` - applies casual mask to the attention matrix
2. `block.py` - it is one transformer block 
3. `ffn.py` - the feed forward layer 
4. `multi_head_attention.py` - applies multi head attention 
5. `pos_embeddings.py` - uses absolute positional encoding (later will evolve to rope)

### step 2 - train a *tiny llm* on a dataset

just to see how transformer works on a small scale apply a tiny llm to a text dataset

*similar to my toy gpt project [link]()*

in here we will use some these tricks too: 
* torch.compile 
* mixed precision training 
* gradient clipping 
* top k/ top p sampling 
* weight decay




