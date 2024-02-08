import logging
from typing import List, Tuple

import transformers
import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from DPR.DPR_module.faiss_indexer import DenseIndexer, DenseFlatIndexer
from DPR.DPR_module.hf_models import get_bert_tensorizer, HFBertEncoder, BertTensorizer


class LocalFaissRetriever:

    def __init__(self, question_encoder: nn.Module, tensorizer: BertTensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, question: List[str]) -> T:
        self.question_encoder.eval()
        with torch.no_grad():
            question = question[0]
            question_token_tensor = [self.tensorizer.text_to_tensor(question)]
            q_ids = torch.stack(question_token_tensor, dim=0).cuda()
            q_seg = torch.zeros_like(q_ids).cuda()
            q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
            _, out, _ = self.question_encoder(q_ids, q_seg, q_attn_mask)
            query_tensor = torch.cat(out.cpu().split(1, dim=0), dim=0)
            return query_tensor

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        results = self.index.search_knn(query_vectors, top_docs)
        return results


class DPR_Retriever:
    def __init__(self):
        ############# Parameters #############
        pretrained_model_cfg = 'DPR/Model'
        sequence_length = 384
        vector_size = 768
        do_lower_case = True
        dropout = 0.1
        ############# Parameters #############
        device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        tensorizer = get_bert_tensorizer(sequence_length, pretrained_model_cfg, do_lower_case)
        question_encoder = HFBertEncoder.init_encoder(
            pretrained_model_cfg,
            projection_dim=0,
            dropout=dropout,
            pretrained=True
        )
        encoder = question_encoder

        encoder.to(device)
        encoder.eval()
        # get questions & answers

        index = DenseFlatIndexer()
        index.init_index(vector_size)
        self.retriever = LocalFaissRetriever(encoder, tensorizer, index)

        # index all passages
        self.retriever.index.deserialize('DPR/DPR_index')

    def get_top_docs_dpr(self, question: str, n_docs: int = 100) -> dict:
        question_tensor = self.retriever.generate_question_vectors([question])

        # get top k results
        top_ids_and_scores = self.retriever.get_top_docs(question_tensor.numpy(), n_docs)
        paragraph_ids = [id.split(':')[1] for id in top_ids_and_scores[0][0]]
        scores = [score for score in top_ids_and_scores[0][1]]
        return dict(zip(paragraph_ids, scores))
