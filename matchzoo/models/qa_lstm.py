import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks


class QALSTM(BaseModel):

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True
        )

        params.add(Param(name='lstm_layers', value=3))
        params.add(Param(name='lstm_hidden_size', value=300))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        self.embedding = self._make_default_embedding_layer()
        self.lstm = nn.LSTM(
            input_size=self._params['embedding_output_dim'],
            hidden_size=self._params['lstm_hidden_size'],
            num_layers=self._params['lstm_layers'],
            bidirectional=True,
            batch_first=True)
        self.dropout = nn.Dropout(p=self._params['dropout_rate'])

    def forward(self, inputs):
        """Forward."""
        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = inputs['text_left'], inputs['text_right']

        # Process left and right input.
        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = self.embedding(input_left.long())
        embed_right = self.embedding(input_right.long())

        self.lstm.flatten_parameters()
        embed_left = self.lstm(embed_left)[0]
        embed_right = self.lstm(embed_right)[0]

        # shape = [B, D]
        embed_left = self.dropout(embed_left.max(dim=1)[0])
        embed_right = self.dropout(embed_right.max(dim=1)[0])

        # # shape = [B, 1]
        # out = F.cosine_similarity(embed_left, embed_right)
        # out = out.unsqueeze(dim=1)
        return embed_right
