import torch
import torch.nn as nn
from model.embedding import GPTEmbedding
from model.transformer import TransformerBlock, LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = GPTEmbedding(cfg["vocab_size"], cfg["emb_dim"], cfg["context_length"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        ) 
        self.final_norm = LayerNorm(cfg["emb_dim"]) 
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        x = self.embedding(in_idx)
        x = self.drop_emb(x)   
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits