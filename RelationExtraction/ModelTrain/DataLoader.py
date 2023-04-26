import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from RelationExtraction.utils import label2rel


# train.txt should be processed by tokenizer first
def tokenize(text, ent_pos):
    # ent_pos [[],[]]
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    encoded = tokenizer(text)
    processed_ent_pos = []
    word_to_token_mapping = []
    for word_id in encoded.words():
        if word_id is not None:
            start, end = encoded.word_to_tokens(word_id)
            word_to_token_mapping.append((start - 1, end - 1))
    word_to_token_mapping = list(dict.fromkeys(word_to_token_mapping))
    for pos in ent_pos:
        start = word_to_token_mapping[pos[0]][0]
        end = word_to_token_mapping[pos[1]][1] - 1
        if start == end:
            processed_ent_pos.append([start])
        else:
            processed_ent_pos.append([i for i in range(start, end + 1)])
    return encoded["input_ids"], encoded['attention_mask'],processed_ent_pos


def collate_fn(batch):
    input_ids = [torch.tensor(f['input_ids']).unsqueeze(0) for f in batch]
    attn = [torch.tensor(f['attention_mask']).unsqueeze(0) for f in batch]
    ent_pos = [f['ent_pos'] for f in batch]
    labels = [f['labels'] for f in batch]
    output = (input_ids, attn, ent_pos, labels)
    return output


def data(file_path_train, file_path_test, batch_size):
    features_train = []
    features_test = []
    with open(file_path_train, encoding="utf-8") as f:
        for i, cont in enumerate(f):
            split = cont.split("\t")
            ent1 = split[1].split(',')
            ent2 = split[2].split(',')
            inputs, attn, ent_pos = tokenize(split[0], [[int(i) for i in ent1], [int(i) for i in ent2]])
            feature = {'input_ids': inputs,
                       'attention_mask':attn,
                       'ent_pos': ent_pos,
                       'labels': label2rel[split[3].replace("\n","")]}
            features_train.append(feature)
    with open(file_path_test, encoding="utf-8") as f:
        for i, cont in enumerate(f):
            ent1 = split[1].split(',')
            ent2 = split[2].split(',')
            inputs, attn, ent_pos = tokenize(split[0], [[int(i) for i in ent1], [int(i) for i in ent2]])
            feature = {'input_ids': inputs,
                       'attention_mask': attn,
                       'ent_pos': ent_pos,
                       'labels': label2rel[split[3].replace("\n", "")]}
            features_test.append(feature)

        # ==============================================================
    #      Use DataLoader to convert to Pytorch acceptable form
    # ==============================================================
    num_classes = len(label2rel)
    train_dataloader = DataLoader(features_train, batch_size=batch_size,
                                         collate_fn=collate_fn, drop_last=False, shuffle=True)
    test_dataloader = DataLoader(features_test, batch_size=batch_size,
                                        collate_fn=collate_fn, drop_last=False)
    return train_dataloader, test_dataloader, num_classes
