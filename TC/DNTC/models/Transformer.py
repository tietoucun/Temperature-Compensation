import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import time

class Model(nn.Module):
    """
    Vanilla Transformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,  attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True,  attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False,  attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.ada_pool = nn.AdaptiveAvgPool1d(configs.seq_len)
        self.ada_pool2 = nn.AdaptiveAvgPool1d(configs.pred_len + configs.label_len)
        self.batchnorm = nn.BatchNorm1d(configs.seq_len)
        self.batchnorm2 = nn.BatchNorm1d(configs.pred_len + configs.label_len)
        self.activ = nn.ReLU()
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        time_now = time.time()
        x_enc = self.activ(self.batchnorm(self.ada_pool(x_enc.permute(0, 2, 1)).transpose(1, 2)))
        x_mark_enc = self.activ(self.batchnorm(self.ada_pool(x_mark_enc.permute(0, 2, 1)).transpose(1, 2)))
        x_dec = self.activ(self.batchnorm2(self.ada_pool2(x_dec.permute(0, 2, 1)).transpose(1, 2)))
        x_mark_dec = self.activ(self.batchnorm2(self.ada_pool2(x_mark_dec.permute(0, 2, 1)).transpose(1, 2)))
        if enc_self_mask is not None:
            enc_self_mask = self.activ(
                self.batchnorm2(self.ada_pool2(enc_self_mask.permute(0, 2, 1)).transpose(1, 2)))

        lowfre_time = time.time() - time_now
        batch_cy = x_dec.clone().detach()
        x_dec = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])], dim=1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :], batch_cy, lowfre_time
