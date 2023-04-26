import torch
import torch.nn as nn
from transformers import AutoModel


class RelationExtractor(nn.Module):
    def __init__(self, args):
        super(RelationExtractor, self).__init__()
        self.bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.rel_classifer = nn.Sequential(
            nn.Linear(2 * 768, 384), nn.LeakyReLU(), nn.Linear(384, 128), nn.LeakyReLU(),
             nn.Linear(128, args["num_rel"]))
        if not args["fine_tune_bert"]:
            for param in self.bert_model.parameters():
                param.requires_grad = False

    def forward(self, inputs, ent_pos):
        """
        a pre-trained BERT is used to learn the embedding of each sentence. relation is learned by attending the
        embedding of each entity. Entities are scored with a logistic layer.
        :param inputs: a list of tokens with the
        BERT special token added. format: [[[idx1, idx2...],[idx1,...]], [[idx1..],[idx3..]]] (
        batch*words)
        :param ent_pos: a list of the positions of entities in each sentence group.
        format: [batch_size*num_ent] e.g., [[0,1],[2,5,6]]
        :return: a list of relation classification results.[cls1,
        cls2, ...]
        """
        encoding, _ = self.bert_model(inputs, return_dict=False)
        emb_list = []
        for i in range(encoding.shape[0]):
            ent_emb_1 = torch.mean(encoding[i][ent_pos[i][0]],dim=0)  # average
            ent_emb_2 = torch.mean(encoding[i][ent_pos[i][1]],dim=0)
            input_emb = torch.cat((ent_emb_1, ent_emb_2), dim=0)
            emb_list.append(input_emb)
        batch_input_emb = torch.stack(emb_list)
        results = self.rel_classifer(batch_input_emb)
        return results

    def inference(self, inputs, ent_pos):
        encoding, _ = self.bert_model(inputs, return_dict=False)
        emb_list = []
        for i in range(encoding.shape[0]):
            ent_emb_1 = torch.mean(encoding[i][ent_pos[i][0]], dim=0)  # average
            ent_emb_2 = torch.mean(encoding[i][ent_pos[i][1]], dim=0)
            input_emb = torch.cat((ent_emb_1, ent_emb_2), dim=0)
            emb_list.append(input_emb)
        batch_input_emb = torch.stack(emb_list)
        results = self.rel_classifer(batch_input_emb)
        predictions = torch.argmax(results, dim=1)
        return predictions


'''
args = {"num_rel":5, "fine_tune_bert":False}
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
sent_list = ["I like studying computer science","I like studying computer science"]
inputs = tokenizer(sent_list, padding=True, truncation=True, return_tensors="pt")
print(inputs)
inputs = inputs["input_ids"]  # num_instances*seq_length(512)
encoding = model.inference(inputs,[[[2,3],[1]],[[2,3],[1]]])
'''