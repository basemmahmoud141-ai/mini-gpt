import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, output_dim, context_length):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        self.pos_embedding_layer = nn.Embedding(context_length, output_dim)
    
    def forward(self, x):
        _, seq_len = x.shape
        token_embeddings = self.token_embedding_layer(x)
        
        positions = torch.arange(seq_len, device=x.device)
        pos_embeddings = self.pos_embedding_layer(positions)
        
        return token_embeddings + pos_embeddings