#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import torch
from torch import Tensor as T
from typing import Tuple
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer


def get_bert_tensorizer(sequence_length, pretrained_model_cfg, do_lower_case, tokenizer=None):
    if not tokenizer:
        tokenizer = get_bert_tokenizer(
            pretrained_model_cfg, do_lower_case=do_lower_case
        )

    return BertTensorizer(tokenizer, sequence_length)


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
            cls,
            cfg_name: str,
            projection_dim: int = 0,
            dropout: float = 0.1,
            pretrained: bool = True,
            **kwargs
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(
                cfg_name, config=cfg, project_dim=projection_dim, **kwargs
            )
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
            self,
            input_ids: T,
            token_type_ids: T,
            attention_mask: T,
            representation_token_pos=0,
    ) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                #return_dict=False
            )
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                #return_dict=False
            )

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                    representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer():
    def __init__(
            self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
            self,
            text: str,
            title: str = None,
            add_special_tokens: bool = True,
            apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                    seq_len - len(token_ids)
            )
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]