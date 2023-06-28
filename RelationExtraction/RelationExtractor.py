from RelationExtraction.Adaptor import PretrainedModel, AdapterModel
import json


class RelationExtractor():
    def __init__(self):
        pretrained_model = PretrainedModel()
        adapter_model = AdapterModel(pretrained_model.config, 3)
        self.re = (pretrained_model, adapter_model)

    def fine_tune_relation_extractor(self):
        from RelationExtraction.ModelTrain.Trainer import Trainer
        from RelationExtraction.ModelTrain.DataLoader import data
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        with open('RelationExtraction/ModelTrain/config.json', 'rb') as f:
            config = json.load(f)
        train_dataloader, test_dataloader = data("RelationExtraction/dataset/example", config["batch_size"], tokenizer)
        trainer = Trainer(self.re, train_dataloader, test_dataloader, **config)
        trainer.train()

    def update_config(self, **kwargs):
        '''
        :param kwargs: example update_config(load_model_path="your path")
        :return: None
        '''
        with open('RelationExtraction/ModelTrain/config.json', 'r') as f:
            config = json.load(f)
        for k in kwargs.keys():
            config[str(k)] = kwargs[k]
        with open('RelationExtraction/ModelTrain/config.json', 'w') as f:
            json.dump(config, f)
