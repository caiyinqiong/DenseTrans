import typing

import torch
import torch.nn as nn
import numpy as np

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks
from matchzoo.utils import parse_activation


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = encoder_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for layer in self.layers:
            layer_out = layer(output, src_mask=mask,
                              src_key_padding_mask=src_key_padding_mask)
            output = torch.cat([output, layer_out], dim=-1)

        if self.norm:
            output = self.norm(output)

        return output


class Dense_Transformer_CLS(BaseModel):

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True
        )
        params.add(Param(name='left_length', value=20))
        params.add(Param(name='right_length', value=100))

        params.add(Param(name='pos_embedding_indim', value=100))
        params.add(Param(name='pos_embedding_outdim', value=300))

        params.add(Param(name='nhead', value=8))
        params.add(Param(name='dim_ffl', value=1024))
        params.add(Param(name='num_layers', value='3'))

        params.add(Param(name='dropout_rate', value=0.1))
        return params

    def build(self):
        self.pos_embedding = nn.Embedding(
            num_embeddings=self._params['pos_embedding_indim'],
            embedding_dim=self._params['pos_embedding_outdim'],
            max_norm=1
        )
        self.word_embedding = self._make_default_embedding_layer()

        d_model = self._params['embedding_output_dim']  # + self._params['pos_embedding_outdim']

        encoder_layers = []
        for i in range(self._params['num_layers']):
            encoder_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model * np.power(2, i),
                nhead=self._params['nhead'],
                dim_feedforward=self._params['dim_ffl'],
                dropout=self._params['dropout_rate']
            ))
        self.encoder = TransformerEncoder(nn.ModuleList(encoder_layers))

        out_feature_size = d_model * np.power(2, self._params['num_layers'])
        self.bilinear = nn.Bilinear(out_feature_size, out_feature_size, 1, True)


    def forward(self, inputs):
        # shape = [B, L]
        # shape = [B, R]
        text_left, text_right = inputs['text_left'], inputs['text_right']

        batch_size = text_left.shape[0]
        pos_left = torch.stack([torch.arange(self._params['left_length'], device=text_left.device)] * batch_size, dim=0)
        pos_right = torch.stack([torch.arange(self._params['right_length'], device=text_left.device)] * batch_size, dim=0)

        mask_left = torch.tensor(np.equal(text_left.cpu().numpy(), 0), device=text_left.device)
        mask_right = torch.tensor(np.equal(text_right.cpu().numpy(), 0), device=text_left.device)

        # shape = [B, L, D]
        # shape = [B, R, D]
        word_embed_left = self.word_embedding(text_left.long())
        word_embed_right = self.word_embedding(text_right.long())
        pos_embed_left = self.pos_embedding(pos_left)
        pos_embed_right = self.pos_embedding(pos_right)

        embed_left = word_embed_left + pos_embed_left
        embed_right = word_embed_right + pos_embed_right

        embed_left = self.encoder(
            embed_left.transpose(0, 1), src_key_padding_mask=mask_left)
        embed_right = self.encoder(
            embed_right.transpose(0, 1), src_key_padding_mask=mask_right)

        # shape = [B, D]
        embed_left = embed_left.transpose(0, 1)[:, 0, :].squeeze(dim=1)
        embed_right = embed_right.transpose(0, 1)[:, 0, :].squeeze(dim=1)

        out = self.bilinear(embed_left, embed_right)

        return out
