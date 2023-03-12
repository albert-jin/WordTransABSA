from dataclasses import dataclass
from typing import List, Optional
import argparse
import logging
from transformers import PreTrainedTokenizer, PreTrainedModel
from functools import reduce
from transformers import DataCollatorForTokenClassification
import torch
import random


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    """
    text: List[str]
    label: Optional[List[str]]


"""
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
"""


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', datefmt="%m/%d/%Y %H:%M:%S"))
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', datefmt="%m/%d/%Y %H:%M:%S"))
        logger.addHandler(stream_handler)
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='twitter', choices=['ACLShortData', 'laprest14', 'rest14', 'lap14',
            'rest15', 'rest16', 'twitter'], help='specify which dataset, please refer to "WordTrans2ABSA/dataset".')
    parser.add_argument("--few_shot",type=int,default=50,choices=[0,5,10,20,50,100],help='0 is default for non-few-shot.')
    parser.add_argument('--log_file', type=str, help='if not specify log_file, default is WordTrans2ABSA/training_logs/{dataset}'
                                                     '/{model_type}@{time}.log.')
    # 模型类型有两种，一种是将集合词取平均，预测最大概率，一种是反向修正和各个类别词汇的空间距离
    parser.add_argument('--model_type', type=str, default='WT2ABSA_Mean', choices=['WT2ABSA_Mean', 'WT2ABSA_Fix'],
                        help='specify which model is adopted.')
    parser.add_argument('--seed', type=int, default=985, help='set seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size, default 16')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate, default 0.0001')
    parser.add_argument('--pre_trained_model', type=str, choices=['pretrainedmodels/bert-base-cased', 't5-base'],
                        default='pretrainedmodels/bert-base-cased')
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--patience', default=5, type=int, help='if {patience} validation not seen increasing, early stop.')
    parser.add_argument('--eval_pre_epochs', type=int, default=1)
    parser.add_argument('--save_best_model', type=bool, default=False)
    parser.add_argument('--answer_strategy', type=str, default='inter_logits_core_token',
                        choices=['most_appear', 'inter_logits_core_token', 'first_appear'])
    args = parser.parse_args()
    return args


def add_labeltokens_to_plm(plm_model: PreTrainedModel, plm_tokenizer: PreTrainedTokenizer, ori_label_token_map):
    """
    add extra label word embeddings to the original PLM tokenizer.
    :param plm_model: Yeah.
    :param plm_tokenizer: Yeah.
    :param ori_label_token_map: {'I-POS': ['good', ...], 'I-NEU': ['objective', ...], 'I-NEG': ['bad', ...]}
    :return:
    """
    added_tokens = list(ori_label_token_map.keys())
    plm_tokenizer.add_tokens(added_tokens)
    num_tokens, _ = plm_model.bert.embeddings.word_embeddings.weight.shape
    plm_model.resize_token_embeddings(num_tokens + len(added_tokens))
    for token in added_tokens:
        label_index = plm_tokenizer.convert_tokens_to_ids(plm_tokenizer.tokenize(token))[0]
        senti_word_indexes = plm_tokenizer.convert_tokens_to_ids(ori_label_token_map[token])
        senti_label_embedding = reduce(lambda x, y: x+y, [plm_model.bert.embeddings.word_embeddings.weight.data[idx]
                                                         for idx in senti_word_indexes])/len(senti_word_indexes)
        plm_model.bert.embeddings.word_embeddings.weight.data[label_index] = senti_label_embedding
    return plm_tokenizer


def add_labeltokens_to_t5(plm_model: PreTrainedModel, plm_tokenizer: PreTrainedTokenizer, ori_label_token_map):
    """
    (For PLM T5) add extra label word embeddings to the original PLM tokenizer.
    :param plm_model: Yeah.
    :param plm_tokenizer: Yeah.
    :param ori_label_token_map: {'I-POS': ['good', ...], 'I-NEU': ['objective', ...], 'I-NEG': ['bad', ...]}
    :return:
    """
    added_tokens = list(ori_label_token_map.keys())
    plm_tokenizer.add_tokens(added_tokens)
    num_tokens, _ = plm_model.shared.weight.shape
    plm_model.resize_token_embeddings(num_tokens + len(added_tokens))
    for token in added_tokens:
        label_index = plm_tokenizer.convert_tokens_to_ids(plm_tokenizer.tokenize(token))[0]
        senti_word_indexes = plm_tokenizer.convert_tokens_to_ids(ori_label_token_map[token])
        senti_label_embedding = reduce(lambda x, y: x+y, [plm_model.shared.weight.data[idx]
                                                         for idx in senti_word_indexes])/len(senti_word_indexes)
        plm_model.shared.weight.data[label_index] = senti_label_embedding
    return plm_tokenizer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def move2cuda(inputs):
    return {k: v.cuda() for k, v in inputs.items()}


class DataCollatorForLMTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        labels = list(map(lambda x: x['labels'], features))
        real_tags = list(map(lambda x: x['real_tags'], features))
        batch = self.tokenizer.pad(features, padding=self.padding, max_length=self.max_length,
                                   pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=None)
        # 因为tokenizer.pad 只将 'token_type_ids' 'input_ids' 'attention_mask' 三个pad, 'labels' 和 'real_tags' 需要手工来pad
        sequence_length = len(batch["input_ids"][0])
        # padding_side = self.tokenizer.padding_side  # padding_side 默认是 "right"
        # label_pad_token_id 默认是-100, 表示该处无需反向优化
        batch['labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        batch['real_tags'] = [real_tag + [self.label_pad_token_id] * (sequence_length - len(real_tag)) for real_tag in real_tags]
        batch = {attr: torch.tensor(batch[attr], dtype=torch.int64) for attr in batch}
        return batch




