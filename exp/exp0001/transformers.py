import math
import torch
from torch import nn
import torch.nn.functional as F


# def pack_seq(
#     seq,
# ):
#     '''

#     長さの異なる系列をbatchで一つにする。その際maskも返す。

#     Parameters:
#     ----------
#     seq: List(torch.tensor)
#         shape = (seq_len, n_landmark, 2)
#         len(seq) = batch_size

#     Returns
#     -------
#     x: torch.tensor
#         shape = (batch_size, max_seq_len, n_landmark*2)
#     x_mask: torch.tensor
#         shape = (batch_size, max_seq_len)

#     '''
#     length = [len(s) for s in seq]
#     batch_size = len(seq)
#     num_landmark = seq[0].shape[1]

#     x = torch.zeros((batch_size, max(length), num_landmark, 2)
#                     ).to(seq[0].device)
#     x_mask = torch.zeros((batch_size, max(length))).to(seq[0].device)
#     for b in range(batch_size):
#         L = length[b]
#         x[b, :L] = seq[b][:L]
#         x_mask[b, L:] = 1
#     x_mask = (x_mask > 0.5)
#     x = x.reshape(batch_size, -1, num_landmark*2)
#     return x, x_mask


class LandmarkEnbedding(nn.Module):
    '''
    LandmarkEnbedding
    '''

    def __init__(self, in_dim, out_dim):
        # input_dim = n_landmarks
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # x.shape = (batch_size, n_landmarks, n_seq)
        # x = x.permute(0, 2, 1)  # (batch_size, n_seq, n_landmarks)
        x = self.mlp(x)  # (batch_size, n_seq, out_dim)
        # x = x.permute(0, 2, 1)  # (batch_size, out_dim, n_seq)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, enbed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, enbed_dim, 2)
                             * (-math.log(10000.0) / enbed_dim))
        pe = torch.zeros(max_len, 1, enbed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EncoderEmbedding(nn.Module):
    '''
    手話のランドマークを埋め込みベクトルに変換
    '''

    def __init__(self, embed_dim, input_size, max_seq_length=512):
        '''
        Parameters
        ----------
        embed_dim: int
            埋め込みベクトルの次元数
        in_dim: int
            入力の次元数
        max_seq_length: int
            入力のフレーム数の最大値
        '''
        super().__init__()
        self.embedding = LandmarkEnbedding(input_size, embed_dim)
        self.position_encoder = PositionalEncoding(
            embed_dim, max_len=max_seq_length)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: torch.tensor
            shape = (batch_size, enc_seq_len, input_size)
            float型のランドマークデータ

        Returns
        -------
        x: torch.tensor
            shape = (batch_size, enc_seq_len, embed_dim)
        '''
        x = self.embedding(x)
        x = self.position_encoder(x)
        return x


class DecoderEmbedding(nn.Module):
    '''
    手話のテキストを埋め込みベクトルに変換
    '''

    def __init__(self, embed_dim, vocab_size=60, max_seq_length=32):
        '''
        Parameters
        ----------
        embed_dim: int
            埋め込みベクトルの次元数
        vocab_size: int
            語彙数
        max_seq_length: int
            文字列の最大長

        '''
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoder = PositionalEncoding(
            embed_dim, max_len=max_seq_length)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        '''
        Parameters
        ----------
        x: torch.tensor
            shape = (batch_size, dec_seq_len)
            long型のテキストデータ, vocab_size未満の整数で構成される

        Returns
        -------
        x: torch.tensor
            shape = (batch_size, dec_seq_len, embed_dim)
        '''
        x = self.embedding(x)
        x = self.position_encoder(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def scaled_dot_product(q, k, v, mask=None):
    # q, k, v: (bs, num_heads, seq_len, head_dim)
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / \
        torch.sqrt(torch.tensor(d_k, dtype=torch.float)
                   )  # (bs, num_heads, seq_len, seq_len)
    if mask is not None:
        # permute(1, 0, 2, 3) => (num_heads, bs, seq_len, seq_len)
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)  # (bs, num_heads, seq_len, head_dim)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()  # (bs, seq_len, d_model)
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length,
                          self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (bs, num_heads, seq_len, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # 3分割 (q, k, v)
        # (bs, num_heads, seq_len, head_dim)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim)  # (bs, seq_len, d_model)
        out = self.linear_layer(values)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask):
        # in practice, this is the same for both languages...so we can technically combine with normal attention
        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length,
                        self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length,
                      self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        # We don't need the mask for cross attention, removing in outer function!
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, max_seq_length, embed_dim, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.encoder_embedding = EncoderEmbedding(
            input_size, embed_dim, max_seq_length)
        self.layers = SequentialEncoder(*[EncoderLayer(embed_dim, ffn_hidden, num_heads, drop_prob)
                                          for _ in range(num_layers)])

    def forward(self, x, self_attention_mask):
        x = self.encoder_embedding(x)
        x = self.layers(x, self_attention_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=embed_dim, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[embed_dim])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(
            d_model=embed_dim, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[embed_dim])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=embed_dim, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[embed_dim])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, embed_dim, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.decoder_embedding = DecoderEmbedding(
            embed_dim, vocab_size, max_seq_length)
        self.layers = SequentialDecoder(
            *[DecoderLayer(embed_dim, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        y = self.decoder_embedding(y)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self,
                 input_size,
                 vocab_size,
                 max_seq_length,
                 embed_dim,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers):
        super().__init__()
        self.encoder = Encoder(input_size, max_seq_length, embed_dim,
                               ffn_hidden, num_heads, drop_prob, num_layers)
        self.decoder = Decoder(vocab_size, max_seq_length, embed_dim,
                               ffn_hidden, num_heads, drop_prob, num_layers)
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self,
                x,
                y,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None):  # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask)
        out = self.decoder(x, y, decoder_self_attention_mask,
                           decoder_cross_attention_mask)
        out = self.linear(out)
        return out
