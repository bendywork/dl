# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/5 23:02
Create User : 19410
Desc : xxx
"""

import torch
from transformers import BertTokenizer

from .. import models, sampling, prediction
from ..entities import Token, Document
from ..input_reader import JsonPredictionInputReader


class Predictor(object):
    def __init__(
            self, model_path, tokenizer_path, types_path, max_span_size,
            size_embedding, rel_filter_threshold
    ):
        super().__init__()
        self.rel_filter_threshold = rel_filter_threshold

        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path,
            do_lower_case=True  # 数据是否转换为小写格式
        )
        self.json_reader = JsonPredictionInputReader(
            types_path, self.tokenizer,
            max_span_size=max_span_size
        )
        self._max_span_size = max_span_size

        self.model = models.load_model(
            'spert', model_path,
            relation_type_count=self.json_reader.relation_type_count,
            entity_type_count=self.json_reader.entity_type_count,
            size_embedding=size_embedding,
            cls_token_id=self.tokenizer.convert_tokens_to_ids('[CLS]'),
            max_pairs=1000,
            prop_drop=0.0,
            freeze_transformer=False
        )
        self.model.eval().cpu()
        print("模型恢复完成!")

    def _parse_tokens(self, jtokens):
        doc_tokens = []

        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self.tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self.tokenizer.encode(token_phrase, add_special_tokens=False)
            if not token_encoding:
                token_encoding = [self.tokenizer.convert_tokens_to_ids('[UNK]')]
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = Token(i, i, span_start, span_end, token_phrase)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [self.tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding

    @torch.no_grad()
    def predict(self, text: str):
        """
        基于给定的文本进行推理预测
        :param text:
        :return:
        """
        # 1. 模型输入数据的构建
        jtokens = text.split(" ")
        doc_tokens, doc_encoding = self._parse_tokens(jtokens)
        document = Document(0, doc_tokens, [], [], doc_encoding)
        batch = sampling.create_eval_sample(document, self._max_span_size)
        for key in batch.keys():
            batch[key] = batch[key][None, ...]  # 扩充维度

        # 2. 调用模型
        result = self.model(
            encodings=batch['encodings'],
            context_masks=batch['context_masks'],
            entity_masks=batch['entity_masks'],
            entity_sizes=batch['entity_sizes'],
            entity_spans=batch['entity_spans'],
            entity_sample_masks=batch['entity_sample_masks'],
            inference=True
        )
        entity_clf, rel_clf, rels = result

        # 3. 转换预测结果
        predictions = prediction.convert_predictions(
            entity_clf, rel_clf, rels,
            batch, self.rel_filter_threshold,
            self.json_reader
        )
        batch_pred_entities, batch_pred_relations = predictions
        final_result = prediction.decoder_result(
            doc=document,
            sample_pred_entities=batch_pred_entities[0],
            sample_pred_relations=batch_pred_relations[0]
        )

        return final_result
