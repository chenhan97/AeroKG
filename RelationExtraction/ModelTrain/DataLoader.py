import torch
from torch.utils.data import DataLoader
from RelationExtraction.utils import label2rel
import numpy as np
import random
import pickle


def collate_fn(features):
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f["input_mask"] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f["segment_ids"] for f in features], dtype=torch.long)
    # if output_mode == "classification":
    all_label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long)
    # elif output_mode == "regression":
    #    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_subj_special_start_ids = torch.tensor([f["subj_special_start_id"] for f in features], dtype=torch.float)
    all_obj_special_start_ids = torch.tensor([f["obj_special_start_id"] for f in features], dtype=torch.float)
    output = (all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_subj_special_start_ids,
              all_obj_special_start_ids)
    return output


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features_trex(examples, max_seq_length,
                                      tokenizer,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      sequence_a_segment_id=0,
                                      mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        text, subj_start, subj_end, obj_start, obj_end, label_id = example[0], example[1], example[2], example[3], \
        example[4], example[5]
        # relation = example.label
        if subj_start < obj_start:
            tokens = tokenizer.tokenize(' '.join(text[:subj_start]))
            subj_special_start = len(tokens)
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text[subj_start:subj_end + 1]))
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text[subj_end + 1:obj_start]))
            obj_special_start = len(tokens)
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text[obj_start:obj_end + 1]))
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text[obj_end + 1:]))
        else:
            tokens = tokenizer.tokenize(' '.join(text[:obj_start]))
            obj_special_start = len(tokens)
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text[obj_start:obj_end + 1]))
            tokens += ['#']
            tokens += tokenizer.tokenize(' '.join(text[obj_end + 1:subj_start]))
            subj_special_start = len(tokens)
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text[subj_start:subj_end + 1]))
            tokens += ['@']
            tokens += tokenizer.tokenize(' '.join(text[subj_end + 1:]))

        _truncate_seq_pair(tokens, [], max_seq_length - 2)
        tokens = ['<s>'] + tokens + ['</s>']
        subj_special_start += 1
        obj_special_start += 1

        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # if output_mode == "classification":
        # label_id = example.label
        # elif output_mode == "regression":
        #    label_id = float(label_map[example.label])

        if subj_special_start >= max_seq_length:
            continue
            # subj_special_start = max_seq_length - 10
        if obj_special_start >= max_seq_length:
            continue
            # obj_special_start = max_seq_length - 10

        subj_special_start_id = np.zeros(max_seq_length)
        obj_special_start_id = np.zeros(max_seq_length)
        subj_special_start_id[subj_special_start] = 1
        obj_special_start_id[obj_special_start] = 1

        features.append({"input_ids": input_ids,
                         "input_mask": input_mask,
                         "segment_ids": segment_ids,
                         "label_id": label_id,
                         "subj_special_start_id": subj_special_start_id,
                         "obj_special_start_id": obj_special_start_id})
    return features


def data(data_path, batch_size, tokenizer):
    with open(data_path, "rb") as fp:
        full_data = pickle.load(fp)
    random.shuffle(full_data)
    # 90% training 10% testing
    training_data = full_data[:int(len(full_data) * 0.9)]
    testing_data = full_data[int(len(full_data) * 0.9):]
    train_features = convert_examples_to_features_trex(training_data, 256, tokenizer,
                                                       # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                       pad_on_left=False,
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                           0],
                                                       pad_token_segment_id=0)
    test_features = convert_examples_to_features_trex(testing_data, 256, tokenizer,
                                                       # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                       pad_on_left=False,
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                           0],
                                                       pad_token_segment_id=0)
    train_dataloader = DataLoader(train_features, batch_size=batch_size,
                                  collate_fn=collate_fn, drop_last=False, shuffle=True)
    test_dataloader = DataLoader(test_features, batch_size=batch_size,
                                  collate_fn=collate_fn, drop_last=False, shuffle=True)
    return train_dataloader, test_dataloader
