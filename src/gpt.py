import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import Normal
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from src.utils import GPTConfig


class CausalSelfAttention(nn.Cell):
    """Causal Self-Attention for language modeling"""

    def __init__(self, config: GPTConfig):
        super(CausalSelfAttention, self).__init__()
        assert config.embedding_size % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Dense(config.embedding_size, config.embedding_size*3)
        # output projection
        self.c_proj = nn.Dense(config.embedding_size, config.embedding_size)
        
        # regularization
        self.attn_dropout = nn.Dropout(keep_prob=config.dropout_rate)
        self.resid_dropout = nn.Dropout(keep_prob=config.dropout_rate)
        self.n_head = config.num_heads
        
        # bias: a lower triangular matrix with 1 and 0 for upper area
        self.mask = Tensor(
            np.tril(np.ones(shape=(config.seq_length, config.seq_length))),
            mstype.float32,
        ).view(1, 1, config.seq_length, config.seq_length) 
        # self.mask = Tensor(
        #     np.triu(np.ones(shape=(config.seq_length, config.seq_length)))
        # ).bool().view(1, 1, config.seq_length, config.seq_length) 

        self.mul = P.BatchMatMul()
        self.mul_t = P.BatchMatMul(transpose_b=True)

    def construct(self, x: Tensor):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensinality
        # print(f"B, T, C: {x.shape}") # -> 128, 128, 512

        q, k, v = F.split(self.c_attn(x), -1, 3)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(0,2,1,3)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(0,2,1,3)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(0,2,1,3)
        # print(f"q,k,v shape: {q.shape}") # -> (128, 8, 128, 64) : (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # q*k/sqrt(d)
        att = self.mul_t(q, k) / F.sqrt(F.scalar_to_tensor(v.shape[-1]))
        # 执行注意力 Mask
        mask = self.mask[:, :, :T, :T]
        att = att * mask + -1e9 * (1 - mask) # much faster than F.masked_fill API
        # att = F.masked_fill(att, mask, -1e9)
        att = P.Softmax(axis=-1)(att)
        att = self.attn_dropout(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = self.mul(att, v)
        # print(f"y shape: {y.shape}")
        y = y.transpose(0,2,1,3).view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Cell):

    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        # causal self-attention layer
        self.attn = CausalSelfAttention(config)
        # feed-forward layer
        self.ln_1 = nn.LayerNorm((config.embedding_size,))
        self.mlp = nn.SequentialCell([
            nn.Dense(config.embedding_size, config.hidden_size),
            nn.ReLU(),
            nn.Dense(config.hidden_size, config.embedding_size),
        ])
        self.ln_2 = nn.LayerNorm((config.embedding_size,))

    def construct(self, x: Tensor):
        x = x + self.attn(x)
        x = self.ln_1(x)

        m = self.mlp(x)
        x = x + m
        x = self.ln_2(x)
        return x


class GPT(nn.Cell):
    def __init__(self, config: GPTConfig):
        super(GPT, self).__init__()
        self.config = config

        self.wte = nn.Embedding(
            config.vocab_size, config.embedding_size, embedding_table=Normal(0.02)
        )
        self.wpe = nn.Embedding(
            config.seq_length, config.embedding_size, embedding_table=Normal(0.02)
        )
        self.dropout = nn.Dropout(keep_prob=config.dropout_rate)
        self.h = nn.SequentialCell([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm((config.embedding_size,))

        self.position_ids = F.arange(config.seq_length)

        self.lm_head = nn.Dense(
            config.embedding_size,
            config.vocab_size,
            weight_init=Normal(0.02),
            has_bias=False,
        )

    def construct(self, idx, targets=None):
        b, t = idx.shape  # batch size, sequence length

        pos = self.position_ids[None, :t]

        x = self.wpe(pos) + self.wte(idx)
        x = self.dropout(x)
        x = self.h(x)
        x = self.ln_f(x)

        if targets is not None:  # 训练阶段
            x = x.view(-1, x.shape[-1])
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1
            )
        else:  # 推理阶段
            x = x[:, [-1], :].view(-1, x.shape[-1])
            logits = self.lm_head(x)  # using list [-1] to preserve the time dim
            loss = None

        return logits, loss


class GPTWithLoss(nn.Cell):
    """
    GPT training loss

    Args:
        network: backbone network of GPT2/3

    Inputs:
        input_ids: the tokenized inputs

    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, network):
        super(GPTWithLoss, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, input_ids):
        # training like a sliding window
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]

        logits, loss = self.network(tokens, labels)

        return loss
