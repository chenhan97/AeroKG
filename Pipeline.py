from NamedEntityRecognition.PLM import NERModel
from RelationExtraction.MLP import RelationExtractor
from RelationExtraction.utils import label2rel, relid2label
from utils import *
import time

class KGConstructor:
    def __init__(self, text_path):
        self.sent_list = []
        self.file_path = text_path
        self.ner = NERModel()
        args = {"num_rel": len(label2rel), "fine_tune_bert": False}
        self.re = RelationExtractor(args)
        self.kg_triplets = []
        #self.read_data()

    def visulization(self):
        return visulization_func(self.kg_triplets)

    def read_data(self):
        with open(self.file_path, encoding="utf-8") as file_reader:
            for i, cont in enumerate(file_reader):
                self.sent_list.append(cont)

    def query_ner(self):
        net_results = self.ner.inference("test if the model loaded")
        if isinstance(net_results, dict):
            if 'error' in net_results.keys():
                time.sleep(net_results['estimated_time']+20)
        a = self.ner.inference(
            "In addition to manufacturing major components for Typhoon, the site builds the aft fuselage and the horizontal and vertical tail planes for every F-35 military aircraft under contract to the prime contractor, Lockheed Martin.")
        print(a)

        for sent in self.sent_list:
            results = self.ner.inference(sent)
            for item in results:
                pass

    def construct(self):
        self.query_ner()

    def fine_tune_relation_extractor(self, num_epoch, lr, batch_size, cuda):
        from RelationExtraction.ModelTrain.Trainer import Trainer
        from RelationExtraction.ModelTrain.DataLoader import data

        train_dataloader, test_dataloader, num_classes = data("RelationExtraction/dataset/train",
                                                              "RelationExtraction/dataset/train", batch_size)
        trainer = Trainer(self.re, num_epoch, lr, train_dataloader, test_dataloader,
                          "RelationExtraction/FineTunedModel/model.pt", cuda)
        trainer.train()
