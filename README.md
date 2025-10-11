# rei-llm 

a large language model, trained from scratch 

from PRE TRAINING -> RLHF 

> [!NOTE]
> i will make this project in parts so all the parts and updates will be in git commit history

### *vanilla transformer*

implemented to revise and see how the transformer works

1. `attn_mask.py` - applies casual mask to the attention matrix
2. `block.py` - it is one transformer block 
3. `ffn.py` - the feed forward layer 
4. `multi_head_attention.py` - applies multi head attention 
5. `pos_embeddings.py` - uses absolute positional encoding (later will evolve to rope)

### train a *tiny llm* on a dataset

just to see how transformer works on a small scale apply a tiny llm to a text dataset

*similar to my toy gpt project [link]()*

in here we will use some these tricks too: 
* torch.compile 
* mixed precision training 
* gradient clipping 
* top k/ top p sampling 
* weight decay

1. `dataset.py` - handles the data loading and batching
2. `eval_loss.py` - evaluates the model's loss on the validation set
3. `gpt.py` - the main gpt model architecture
4. `run_tinyllm.py` - a script to run the whole pipeline
5. `sample.py` - generates text from the trained model
6. `tiny_hi.txt` - the dataset used for training
7. `tok.py` - a simple character-level tokenizer
8. `train.py` - the training loop for the model
9. `utils.py` - utility functions - topk and topp sampling

see the output of the run here - [output](tiny-llm/out.txt)

### the *modern* tranformer

implemted a modern version of the [vanilla-transformer](vanilla-transformer) 

includes modern stuff like 
* rope
* rms norm 
* swiglu 
* kv cache 
* sliding window attention 



