import math
import pandas as pd

import torch
import torch.nn as nn
from d2l import torch as d2l


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    X = X.permute(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 `transpose_qkv` 的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])

    X = X.permute(0, 2, 1, 3)

    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    # input/output: batch_size * seq_len * input_dim
    # q
    # k
    # v
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)

        # nn.Linear(in_features, output_features)
        # W_q, W_k, W_v拆成nums_head份对应不同的权值
        # 可视为所有head里面的W_q, W_k, W_v分别叠在一起
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        print('size of queries: ', queries.shape)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        print('size of keys: ', queries.shape)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        print('size of values: ', queries.shape)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # 实际上MLP
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # residual = self.dropout(Y) + X
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)

        self.blks = nn.Sequential()
        for i in range(num_layers):
            # num_layers: # of encoder blk
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                             dropout, use_bias)
            )

    def forward(self, X, valid_lens, *args):
        # X: token
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)

        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention.attention_weights
            return X


class DecoderBlock(nn.Module):
    """ i-th decoder """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, use_bias=False, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                             num_heads, dropout, use_bias)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]



def test_attention(num_hiddens, num_heads):
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries, num_kvparis, valid_lens = 2, 4, 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_queries, num_hiddens)) # [2, 4, 100]

    return attention(X, Y, Y, valid_lens).shape


def test_ffn(ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
    ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_outputs)
    ffn.eval()
    return ffn(torch.ones((2, 3, 4))).shape


def test_norm():
    ln = nn.LayerNorm(2)
    bn = nn.BatchNorm1d(2)
    X = torch.tensor([[1, 2],
                      [2, 3]], dtype=torch.float32)
    print('layer norm: ', ln(X))
    print('batch norm: ', bn(X))


def test_addnorm():
    # 不会改变输入输出形状
    add_norm = AddNorm([3, 4], 0.5)
    # add_norm = AddNorm([3, 4], 0.5)
    add_norm.eval()
    return add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape


def test_encoder_blk():
    # 不会改变输出形状
    X = torch.ones((2, 100, 24))
    valid_lens = torch.tensor([3, 2])
    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    return encoder_blk(X, valid_lens).shape


def test_encoder():
    # 2 layers
    encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
    encoder.eval()
    return encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape


if __name__ == '__main__':
    # print(test_attention(100, 5))
    # print(test_ffn(4, 4, 8))  # [2, 3, 4] -> [2, 3, 8]
    # test_norm()
    # test_norm()
    print(test_addnorm())
    #  print(test_encoder_blk())
