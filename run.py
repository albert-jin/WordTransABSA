from utils import get_args, set_logger, add_labeltokens_to_plm, add_labeltokens_to_t5, DataCollatorForLMTokenClassification, set_seed, move2cuda
from datetime import datetime
import pathlib
from collections import OrderedDict
from tqdm import tqdm
from datasets import load_dataset
from functools import reduce
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, SchedulerType, get_scheduler, T5TokenizerFast, T5ForConditionalGeneration, T5Config
from torch.utils.data.dataloader import DataLoader
from sklearn import metrics
import torch
import random


def main():
    args = get_args()
    set_seed(args.seed)
    # if not specify log_file, default is WordTrans2ABSA/training_logs/{dataset}/{model_type}@{time}.log.
    log_file = f'../WordTrans2ABSA/training_logs/{args.dataset}/{args.model_type}@{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.log'
    if args.few_shot!=0:
        log_file = f'../WordTrans2ABSA/training_logs/{args.dataset}_fewshot@{args.few_shot}/{args.model_type}@{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}.log'
    filepath = pathlib.Path(log_file)
    father_dir = filepath.parent
    father_dir.mkdir(parents=True, exist_ok=True)
    logger = set_logger(args.log_file if args.log_file else log_file)
    logger.info('start.')
    logger.info('--------hyper-arguments for this training.--------')
    logger.info(args)
    data_path_prefix = '../WordTrans2ABSA/dataset/{}/{}.json'
    data_files = {'train': data_path_prefix.format(args.dataset, 'train'), 'dev': data_path_prefix.format(args.dataset,
                  'dev'), 'test': data_path_prefix.format(args.dataset, 'test')}
    logger.info('----------dataset files------------')
    logger.info(data_files)
    raw_datasets = load_dataset('json', data_files=data_files)
    column_names = raw_datasets["train"].column_names
    text_column_name, label_column_name = column_names
    ori_label_token_map = {
        'I-POS': ['good', 'clever', 'excellent', 'effectual', 'beautiful', 'nice', 'useful', 'well', 'fantastic'],
        'I-NEU': ["Michael", "John", "David", "Thomas", "Martin", "Paul"],
        'I-NEG': ['bad', 'terrible', 'awful', 'badly', 'terrible', 'poor', 'unfortunate', 'harmful', 'dirty', 'sorry']}
    logger.info('-----------------------------------')
    logger.info(ori_label_token_map)
    labels_list = list(ori_label_token_map.keys())
    label2id, id2label = {'O': 0}, {0: 'O'}
    for label in labels_list:
        label2id[label] = len(label2id)
        id2label[len(id2label)] = label
    logger.info('-----------------------------------')
    logger.info(label2id)
    logger.info(id2label)
    if args.pre_trained_model == 'pretrainedmodels/bert-base-cased':
        plm_config = AutoConfig.from_pretrained(args.pre_trained_model)
        plm_tokenizer = AutoTokenizer.from_pretrained(args.pre_trained_model, use_fast=True, do_lower_case=False)
        plm_model = AutoModelForMaskedLM.from_pretrained(args.pre_trained_model, config=plm_config)  # from_tf= False
    elif args.pre_trained_model == 't5-base':
        plm_config = T5Config.from_pretrained(args.pre_trained_model)
        plm_tokenizer = T5TokenizerFast.from_pretrained(args.pre_trained_model)
        plm_model = T5ForConditionalGeneration.from_pretrained(args.pre_trained_model, config=plm_config)  # from_tf= False
    else:
        logger.info(f'{args.pre_trained_model} not in the specific pre-trained models.')
        return
    plm_model = plm_model.cuda()
    plm_model.train()
    if args.model_type == 'WT2ABSA_Mean':  # 将新的平均类别字加入PLM的字典中
        if args.pre_trained_model == 'pretrainedmodels/bert-base-cased':
            plm_tokenizer = add_labeltokens_to_plm(plm_model, plm_tokenizer, ori_label_token_map)
        else:
            plm_tokenizer = add_labeltokens_to_t5(plm_model, plm_tokenizer, ori_label_token_map)
        label2tokenid = {label: plm_tokenizer.convert_tokens_to_ids(label) for label in labels_list}
        tokenid2label = {label2tokenid[x]: x for x in label2tokenid}
        logger.info('-----------------------------------')
        logger.info(label2tokenid)
        logger.info(tokenid2label)

    def preprocess_dataset(examples):
        """
        :param examples: tokenize 后的句子数组
        :return: 产出PLM的输出标签 all_labels 和真实标签 all_real_tags
        all_labels [[-100,0,0,0,1,1,0,0,0,0,-100],...]
        all_real_tags [[-100,382,129,528,28996, 28996, 7821,3471,2349,5637,-100],...]
        """
        # 默认情感词的每个word和他的拆分词都要给标签
        tokenized_inputs = plm_tokenizer(examples[text_column_name], max_length=128, padding=False, truncation=True, is_split_into_words=True)
        all_labels = []  # PLM应该输出的标签
        all_real_tags = []  # 解析PLM标签后的真实类别标签
        for idx, ori_labels in enumerate(examples[label_column_name]):
            # [101, 5651, 4601, 1166, 151, 119, 2270, 1739, 1525, 119, 102]  151 和 119是一个单词拆分后的
            input_ids = tokenized_inputs.input_ids[idx]
            word_ids = tokenized_inputs.word_ids(batch_index=idx)  # [None, 0, 1, 2, 3, 3, 4, 5, 6, 7, None]
            temp_labels = ori_labels + ['O']
            if args.pre_trained_model == 'pretrainedmodels/bert-base-cased':
                temp_labels = ['O'] + temp_labels
            _previous_word_id = -1
            for _idx, _word_id in enumerate(word_ids):
                if _previous_word_id == _word_id:  # 需要在ori_labels补位
                    temp_labels.insert(_idx, temp_labels[_idx])
                _previous_word_id = _word_id
            assert len(temp_labels) == len(word_ids) == len(input_ids)
            example_tags = [label2id[_label] for _label in temp_labels]
            # 通过 temp_labels, input_ids, word_ids 制造 PLM 的直接输出: example_labels
            example_labels = []
            for _input_id, _label, _word_id in zip(input_ids, temp_labels, word_ids):
                if _word_id is None:
                    example_labels.append(-100)
                else:
                    if _label == 'O':  # 普通标签下，PLM的输出不做变换
                        example_labels.append(_input_id)  # example_labels.append(-100)  发现效果真不太行
                    else:  # 标记标签需要变换成新类别标签
                        example_labels.append(label2tokenid[_label])
            assert len(example_tags) == len(example_labels)
            all_labels.append(example_labels)
            all_real_tags.append(example_tags)
            tokenized_inputs['labels'] = all_labels
            tokenized_inputs['real_tags'] = all_real_tags
        return tokenized_inputs

    logger.info('start preprocess dataset.')
    processed_raw_datasets = raw_datasets.map(
        preprocess_dataset,
        batched=True,
        remove_columns=column_names,
        desc="preprocessing dataset, including tokenizer, aligning and packing",
    )
    logger.info('dataset preprocess finished.')

    #
    stats = {28996:0,28997:0,28998:0}
    def filterfewshot(example, idx):
        for key in stats:
            if key in example['labels']:
                if stats[key]<args.few_shot:
                    if random.random() < 0.5:  # 使得每次采样都具有一定随机性
                        stats[key]+=1
                        return True
                    else:
                        return False
                else:
                    return False
    #
    train_dataset_temp, dev_dataset, test_dataset = processed_raw_datasets['train'], processed_raw_datasets['dev'], processed_raw_datasets['test']
    if args.few_shot !=0:
        train_dataset = train_dataset_temp.filter(filterfewshot, with_indices=True) # lambda example, idx: idx < args.few_shot
    else:
        train_dataset = train_dataset_temp
    logger.info(f'Train: {train_dataset.shape[0]}, Dev: {dev_dataset.shape[0]}, Test: {test_dataset.shape[0]}.')

    # 训练数据集准备.
    data_collator = DataCollatorForLMTokenClassification(plm_tokenizer, pad_to_multiple_of=None)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    logger.info('dataloader prepare finished.')

    def evaluate(mode='dev'):  # test
        """
        NEED: label2id, id2label, label2tokenid, tokenid2label,
        :param mode: 
        :return: 
        """
        plm_model.eval()
        all_pred_tags = []
        all_real_tags = []
        situation_record = {'位置第一个显示标签': 0, '其他位置有情感标签': 0, '全句没有情感标签': 0}
        # 创建从28996 到1,2,3 标签的映射
        tokenid2id = []
        for _label in label2tokenid:
            # if _label in label2id:  # _label保证在label2id 里
            tokenid2id.append((label2tokenid[_label], label2id[_label]))
        index_tensor = torch.tensor(list(map(lambda x: x[0], tokenid2id)), dtype=torch.int64, device=torch.device("cuda:0"))
        logger.info(index_tensor)
        for _step, _batch_inputs in enumerate(tqdm(dev_dataloader if mode == 'dev' else test_dataloader)):
            with torch.no_grad():
                # [0,0,0,0,0,1,1,0,0,0,0,0]
                real_tags = _batch_inputs.pop('real_tags', 'not found real_tags')
                _batch_inputs = move2cuda(_batch_inputs)
                eval_outputs = plm_model(**_batch_inputs, output_hidden_states=True)
            predictions = eval_outputs.logits.argmax(dim=-1)  # (batch_size, seq_len)
            pred_tags = predictions.detach().cpu().clone().numpy()
            real_tags = real_tags.detach().cpu().clone().numpy()
            assert pred_tags.shape == real_tags.shape
            # 先通过寻找real_tags 第一个非零元素确定情感极性, next寻找pred_tags在对应位置的输出解码是否非零, 如果非零,
            # 则通过二者判断是否正确, 如果第一个是零,则寻找第一个出现的(改为统计里面最多出现的)情感标签作为该句子的情感,
            # 如果整句中没有情感标签, 则默认中性 I-NEU.
            for sent_idx, (pred_tag, real_tag) in enumerate(zip(pred_tags, real_tags)):
                for token_idx, (p_tag, r_tag) in enumerate(zip(pred_tag, real_tag)):
                    if r_tag != 0:
                        all_real_tags.append(r_tag)  # 真实情感极性tag.
                        if p_tag in tokenid2label:
                            final_tag = label2id[tokenid2label[p_tag]]
                            all_pred_tags.append(final_tag)
                            situation_record['位置第一个显示标签'] += 1
                        else:
                            if args.answer_strategy == 'inter_logits_core_token':
                                # 找当前位置的情感标签集合中拥有大概率的对应情感标签
                                all_probs = eval_outputs.logits[sent_idx, token_idx]
                                pred_senti_probs = torch.index_select(all_probs, 0, index_tensor).detach().cpu().tolist()
                                # logger.info(pred_senti_probs)
                                max_senti_prob = -999
                                final_tag = 0
                                for senti_prob, sub_tokenid2id in zip(pred_senti_probs, tokenid2id):
                                    if senti_prob > max_senti_prob:
                                        max_senti_prob = senti_prob
                                        final_tag = sub_tokenid2id[1]
                                if not final_tag > 0:
                                    logger.error(f'final_tag = 0: pred_senti_probs: {pred_senti_probs} tokenid2id: {tokenid2id}')
                                all_pred_tags.append(final_tag)
                            else:
                                # most_appear and first_appear
                                # 如果当前位置不是情感标签,是普通字符. (r_tag 指代 真实情感极性)
                                if args.answer_strategy == "most_appear":
                                    senti_stats = {}
                                    for p_t in pred_tag:
                                        if p_t in tokenid2label:
                                            if p_t not in senti_stats:
                                                senti_stats[p_t] = 0
                                            else:
                                                senti_stats[p_t] += 1
                                    if len(senti_stats) != 0:  # 在句子找到至少一个情感极性标签
                                        situation_record['其他位置有情感标签'] += 1
                                        final_temp_tag = reduce(lambda x, y: x if x[1] > y[1]
                                        else y, senti_stats.items())[0]  # 属于28996 这种
                                        final_tag = label2id[tokenid2label[final_temp_tag]]
                                        all_pred_tags.append(final_tag)
                                    else:  # 没找到任何情感标签的话, 就默认中性 I-NEU.
                                        situation_record['全句没有情感标签'] += 1
                                        all_pred_tags.append(label2id['I-NEU'])
                                elif args.answer_strategy == 'first_appear':
                                    final_tag = None
                                    for p_t in pred_tag:
                                        if p_t in tokenid2label:
                                            final_tag = label2id[tokenid2label[p_t]]
                                            break
                                    final_tag = final_tag if final_tag else label2id['I-NEU']
                                    all_pred_tags.append(final_tag)
                        break
        assert len(all_pred_tags) == len(all_real_tags)
        logger.info('**********Results Comparison**********')
        logger.info(f'真实标签: \n {all_real_tags}')
        logger.info(f'预测标签: \n {all_pred_tags}')
        logger.info('---------Evaluated Metrics----------')
        accuracy = metrics.accuracy_score(all_real_tags, all_pred_tags)
        precision = metrics.precision_score(all_real_tags, all_pred_tags, labels=[1, 2, 3], average='macro')
        recall = metrics.recall_score(all_real_tags, all_pred_tags, labels=[1, 2, 3], average='macro')
        f1_score = metrics.f1_score(all_real_tags, all_pred_tags, labels=[1, 2, 3], average='macro')
        logger.info(f'Accuracy: {accuracy}.')
        logger.info(f'Macro Precision: {precision}.')
        logger.info(f'Macro Recall: {recall}.')
        logger.info(f'Macro F1 score: {f1_score}.')
        logger.info('---------confusion_matrix----------')
        confusion_matrix = metrics.confusion_matrix(all_real_tags, all_pred_tags, labels=[1, 2, 3])
        logger.info(f'\n{confusion_matrix}')
        logger.info(f'模型效果验证详情: {situation_record}.')
        plm_model.train()
        return accuracy, precision, recall, f1_score

    # 训练要素准备.
    optimizer_grouped_parameters = [{"params": [p for _, p in plm_model.named_parameters()], "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    train_steps = args.total_epochs * len(train_dataloader)  # 注意: 此时dataloader 长度是 dataset/batch_size
    lr_scheduler = get_scheduler(SchedulerType.COSINE, optimizer, 0, train_steps)
    logger.info("***** Start Training *****")
    logger.info(f'Total Epochs: {args.total_epochs}')
    logger.info(f'validate on per-{args.eval_pre_epochs} epochs.')
    logger.info(f'Training Samples Count: {train_dataset.shape[0]}.')
    logger.info(f'Batch Size: {args.batch_size}, 每轮 {len(train_dataloader)} Batch.')
    logger.info(f'Learning Rate: {args.learning_rate}')

    best_accuracy, best_precision, best_recall, best_f1_score, patience = 0, 0, 0, 0, 0
    # 准确率单独, best_precision, best_recall是best_f1_score下的对应值
    for epoch_idx in range(args.total_epochs):
        logger.info(f'--------------Epoch. {epoch_idx+1}.---------------')
        tot_loss = 0
        prog_bar = tqdm(train_dataloader, desc=f'[Epoch. {epoch_idx + 1}.]')
        for step, batch_inputs in enumerate(prog_bar):
            _ = batch_inputs.pop('real_tags', 'not found real_tags')
            batch_inputs = move2cuda(batch_inputs)
            outputs = plm_model(**batch_inputs)
            # outputs: .loss (一个标量) .logits 序列中每一个词的概率 维度: (batch_size, seq_len, vocab_size) 16,44, 29000
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=plm_model.parameters(), max_norm=10, norm_type=2)
            tot_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            prog_bar.set_postfix(ordered_dict=OrderedDict(avg_loss=tot_loss / (step + 1)))
        # 保存进度条
        logger.info(prog_bar)
        logger.info("Epoch {}, average loss: {}".format(epoch_idx + 1, tot_loss / len(train_dataloader)))
        # 验证集验证
        if epoch_idx % args.eval_pre_epochs == 0:
            logger.info(f'--------------Evaluate on Epoch. {epoch_idx + 1}.---------------')
            accuracy, precision, recall, f1_score = evaluate('dev')
            patience += 1
            if accuracy > best_accuracy:
                patience = 0
                best_accuracy = accuracy
                if args.save_best_model:
                    save_path = f'../WordTrans2ABSA/state_dict/{args.model_type}@{args.dataset}_acc_{round(best_accuracy, 4)}'
                    filepath = pathlib.Path(save_path)
                    father_dir = filepath.parent
                    father_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(plm_model.state_dict(), save_path)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_precision = precision
                best_recall = recall
            logger.info(f'----------------------Evaluate finished.------------------------')
            if patience > args.patience:
                logger.info(f'Based on [{epoch_idx+1-args.patience}.th Epoch] best_accuracy: {best_accuracy}.')
                logger.info(f'latest {args.patience} epochs not seen performance increased, early stop.')
                break
    logger.info('********** Best Results on Dev Dataset **********')
    logger.info(f'Best Accuracy: {best_accuracy}.')
    logger.info(f'Best Macro Precision: {best_precision}.')
    logger.info(f'Best Macro Recall: {best_recall}.')
    logger.info(f'Best Macro F1 score: {best_f1_score}.')
    logger.info("***** Training Over *****")
    logger.info(f'--------------Testing ...---------------')
    evaluate('test')
    logger.info(f'----------------------Testing finished.------------------------')
    logger.info('bengio.')


if __name__ == '__main__':
    main()
