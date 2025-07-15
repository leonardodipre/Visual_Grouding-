import torch
import torch.nn as nn
import math
from util import compute_softmax_mean_heads_attention


###### Positional Encoding ######

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    # Create a matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)

    # Create a vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    # Denominator of the formula
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Apply the sin to even positions
    pe[:, 0::2] = torch.sin(position * div_term)

    # and the cos to odd positions
    pe[:, 1::2] = torch.cos(position * div_term) # [:, 1::2] -> start from 1 and move with step=2

    pe = pe.unsqueeze(0) # (1, seq_len, d_model) -> for batch

    self.register_buffer('pe', pe) # we do this when we have a tensor that we wanna keep inside the model,
    # when saving the model we will also save this.


  def forward(self, x):
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # not learned
    return self.dropout(x)


############ ENCODER ############


###### Multi-Head Attention ######

class MultiHeadAttentionBlock(nn.Module):

  def __init__(self, d_model: int, heads: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.h = heads
    assert d_model % heads == 0, 'd_model not divisible by h'
    self.d_k = d_model // heads # // for having int, d_k = d_v = d_model/heads
    self.w_q = nn.Linear(d_model, d_model) # Wq
    self.w_k = nn.Linear(d_model, d_model) # Wk
    self.w_v = nn.Linear(d_model, d_model) # Wv
    self.w_0 = nn.Linear(d_model, d_model) # W0 -> (heads*d_v, d_model) -> (d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  @staticmethod # no need for instance of this class to call this method
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k =  query.shape[-1]
    # (batch_size, h, d_k) -> (batch_size, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # @ -> matrix multiplication
    # Before the softmax we wanna mask some values (future tokens or padding tokens to interact with each other...)
    if mask is not None:
      # where is True substitute with very big, negative number
      attention_scores.masked_fill(mask == 0, -1e9)
    
    # In the paper of AttBalance they first perform the mean and then the softmax...
    attention_scores_without_softmax = attention_scores

    attention_scores = attention_scores.softmax(dim = -1) # (batch_size, h, seq_len, seq_len)

    if dropout is not None:
      attention_scores = dropout(attention_scores)

    return (attention_scores @ value), attention_scores_without_softmax


  def forward(self, q, k, v, mask):
    # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model) for all the following three
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)
    """
      Now we want to divide these matrixes in heads
      we want to split the embedding of the sentence, not the sentence
      keep the batch dimension, keep the sentence, split the embedding
    """
    # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

    # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k) #.contiguos() returns a contiguous in memory tensor containing the same data as self tensor

    # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
    return self.w_0(x)


###### Add & Norm ######

class LayerNormalization(nn.Module):

  def __init__(self, eps:float = 10e-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
    self.bias = nn.Parameter(torch.zeros(1)) # Added


  def forward(self, x):
    mean = x.mean(dim = -1, keepdim=True)
    std = x.std(dim = -1, keepdim=True)
    return (x - mean) / (std + self.eps) * self.alpha + self.bias


class AddAndNormBlock(nn.Module):

  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization() # or nn.LayerNorm()


  def forward(self, x, sublayer_x):
    # sublayer_x -> previous layer's x [to implement the residual connection]
    return self.dropout(self.norm((x + sublayer_x)))
  

  ###### Feed Forward ######

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2


  def forward(self, x):
    # input shape (batch, seq_len, d_model)
    # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
    x = self.linear_1(x)
    x = torch.relu(x) # max (0, xW1 + b1)
    x = self.dropout(x)
    # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
    x = self.linear_2(x)
    return x


###### Encoder Block ######
"""
x  ->  [self-attention]  ->  [residual + norm]  ->  [feed-forward]  ->  [residual + norm]
"""

class EncoderBlock(nn.Module):
  def __init__(self, self_attention_block: nn.MultiheadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
    super().__init__()
    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    self.add_norm_1 = AddAndNormBlock(dropout)
    self.add_norm_2 = AddAndNormBlock(dropout)

  def forward(self, x, mask):
    # mask for padding tokens.
    # we are calling the self_attention_block with the same input (Self attention, won't be the same for Cross attention)
    self_attention_block_output = self.self_attention_block(x, x, x, mask)

    # For AttBalance
    attention_maps = self.self_attention_block.attention_scores
    
    x = self.add_norm_1(x, self_attention_block_output)

    feed_forward_block_output = self.feed_forward_block(x)
    
    x = self.add_norm_2(x, feed_forward_block_output)

    return x, attention_maps



###### Encoder (Nx Encoder Block) ######
class Encoder(nn.Module):

  def __init__(self, layers: nn.ModuleList):
    super().__init__()
    self.layers = layers # how many encoder layers (blocks) we stacked

  def forward(self, x, mask):
    all_attention_maps = []
    for layer in self.layers:
      x, attention_map = layer(x, mask)
      softmaxed_mean_attention = compute_softmax_mean_heads_attention(attention_map)
      all_attention_maps.append(softmaxed_mean_attention)
    return x, all_attention_maps


def build_encoder_stack(num_encoders, d_model, h, dropout, d_ff):
    # Create the encoder blocks
    encoder_blocks = []
    
    for i in range(num_encoders):
      encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
      encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    return encoder