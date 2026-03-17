import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        #Q, K, V matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model, n_heads)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.head_dim)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.head_dim)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.head_dim)
        
        # transpose for attention calculation
        # (batch_size, n_heads, seq_len, head_dim)
        Q, K, V = [x.transpose(1, 2) for x in (Q, K, V)]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim) #（batch_size, n_head, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) #若使用float('-inf')，则softmax后会得到NaN，因此使用一个非常小的数来代替-inf
            
        # softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # weighted sum of values
        context = torch.matmul(attn_weights, V) # (batch, n_heads, seq, head_dim)
        
        # concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(context)
        return output, attn_weights
    
class FFN(nn.Module):
    def __init__(self, d_model =512, d_ff=2048, activation=nn.ReLU, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1,activation=nn.ReLU):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN(d_model, d_ff, activation, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        
        return x

def get_sinusoidal_position_encoding(d_model=512, max_len=5000):
    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0) # (1, max_len, d_model)
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return x
    return SinusoidalPositionalEncoding(d_model, max_len)

class Encoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, max_len=5000, activation=nn.ReLU):
        super().__init__()
        self.embeddings = nn.Embedding(10000, d_model)
        self.d_model = d_model
        self.pos_encoder = get_sinusoidal_position_encoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        x = self.embeddings(src) * math.sqrt(self.d_model)  # scale by sqrt(d_model) to stabilize gradients      
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
            
        return x
        

# encoder = Encoder()
# x = torch.randint(0, 1000, (2, 10))
# output = encoder(x)
# 
# print(f"输入 shape: {x.shape}")
# print(f"输出 shape: {output.shape}")
# print(f"参数量: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")
    

pos_encoder = get_sinusoidal_position_encoding(d_model=64, max_len=100)
pe = pos_encoder.pe.squeeze(0).numpy()
plt.figure(figsize=(12, 6))
plt.imshow(pe, cmap='RdBu', aspect='auto')
plt.colorbar()
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Position Encoding')
plt.show()