"""
Transfer the original dataset from https://github.com/albert-jin/DictionaryFused-E2E-ABSA to a standard format which
 would be fed into the MODEL.
"""

import os
import csv
import json


# "label" "text"
all_err_count = 0
for filename in os.listdir('.'):
    if filename not in ['rest14', 'lap14']:
        continue
    filepath = os.path.join('.', filename)
    if os.path.isdir(filepath):
        for sub_filename in os.listdir(filepath):
            file = os.path.join(filepath, sub_filename)
            if os.path.isfile(file) and sub_filename.endswith('tsv'):
                new_file = os.path.join(filepath, sub_filename.replace('.tsv', '_io.json'))
                used_file = os.path.join(filepath, sub_filename.replace('tsv', 'json'))
                with open(file, mode='rt', encoding='utf-8') as inp, open(new_file, mode='wt', encoding='utf-8') as \
                    outp, open(used_file, mode='wt', encoding='utf-8') as used_outp:
                    reader = csv.reader(inp, delimiter='\t')
                    notin_count, err_count = 0, 0
                    for idx, row in enumerate(reader):
                        if len(row) != 3 or idx == 0:
                            continue
                        sentence, target, tag = row
                        start_idx = sentence.find(target)
                        if start_idx != -1:
                            if start_idx != 0 and sentence[start_idx-1] != " ":
                                bef_sent = sentence[:start_idx] + ' '
                            else:
                                bef_sent = sentence[:start_idx]
                            if (start_idx + len(target) != len(sentence)) and sentence[start_idx+len(target)] != " ":
                                aft_sent = ' ' + sentence[start_idx+len(target):]
                            else:
                                aft_sent = sentence[start_idx+len(target):]
                            sentence = bef_sent + target + aft_sent
                        else:
                            notin_count += 1
                            all_err_count += 1
                            # print(sentence,'@', target)
                            continue
                        TAG = "NEU" if tag == '0' else ("NEG" if tag == '-1' else "POS")
                        tokens = list(filter(lambda x: x != '', sentence.split(' ')))
                        sub_tokens = list(filter(lambda x: x != '', target.split(' ')))
                        labels_list = len(tokens) * ['O']
                        target_len = len(sub_tokens)
                        SUCC = False
                        for i in range(0, len(tokens)-target_len+1):
                            sub_sent = tokens[i: i+target_len]
                            if sub_sent == sub_tokens:
                                labels_list[i: i + target_len] = ['I-'+TAG] * target_len
                                SUCC = True
                                break
                        if not SUCC:
                            err_count += 1
                            all_err_count += 1
                            # print(tokens,'@', sub_tokens)
                        else:
                            json_row = {"text": tokens, "label": labels_list}
                            outp.write(json.dumps(json_row, ensure_ascii=False)+'\n')
                            used_outp.write(json.dumps(json_row, ensure_ascii=False)+'\n')
                    print('数据集不对数:', notin_count, '处理出错数:', err_count)
print('acl,twitter, rest14,15,16 五个数据集总共预处理出错数:', all_err_count)
