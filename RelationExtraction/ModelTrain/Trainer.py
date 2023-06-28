import time
from sklearn.metrics import f1_score
import torch
from transformers import get_linear_schedule_with_warmup
import os

class Trainer:
    def __init__(self, models, train_dataloader, test_dataloader, **kwargs):
        self.cuda = kwargs["cuda"]
        self.warmup_steps = kwargs["warmup_steps"]
        self.epochs = kwargs["epochs"]
        self.learning_rate = kwargs["learning_rate"]
        self.load_model_path = kwargs["load_model_path"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.adam_epsilon = kwargs["adam_epsilon"]
        self.weight_decay = kwargs["weight_decay"]
        self.max_grad_norm = kwargs["max_grad_norm"]
        self.pretrained_model = models[0]
        self.adapter_model = models[1]
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.best_epoch = 0
        self.best_f1 = 0.0

    def train(self):
        total_t0 = time.time()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.adapter_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.adapter_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps =len(self.train_dataloader) // self.gradient_accumulation_steps * self.epochs)

        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            self.total_train_loss = 0.0
            accum_iter = 1
            for step, batch in enumerate(self.train_dataloader):
                self.pretrained_model.eval()
                self.adapter_model.train()
                inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1].to(self.cuda),
                              'token_type_ids': None,
                              # XLM and RoBERTa don't use segment_ids
                              'labels': batch[3].to(self.cuda),
                              'subj_special_start_id': batch[4].to(self.cuda),
                              'obj_special_start_id': batch[5].to(self.cuda)}
                pretrained_model_outputs = self.pretrained_model(**inputs)
                outputs = self.adapter_model(pretrained_model_outputs, **inputs)
                loss = outputs[0]
                self.total_train_loss += loss.item()

                # normalize loss to account for batch accumulation
                loss = loss / accum_iter
                # backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.adapter_model.parameters(), self.max_grad_norm)

                # weights update
                if ((step + 1) % accum_iter == 0) or (step + 1 == len(self.train_dataloader)):
                    self.scheduler.step()
                    self.optimizer.step()
                    self.pretrained_model.zero_grad()
                    self.adapter_model.zero_grad()

                if step % 1 == 0:
                    f1 = self.evaluate()
                    print("F1 score on testing set: ", f1)
                    self.adapter_model.train()
            print("")
            print("  Total training loss: {0:.2f}".format(self.total_train_loss))

        print("")
        print("Training complete!")
        return None

    def evaluate(self):
        # ========================================
        #             Validation / Test
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        # Also applicable to test set.

        dataloader = self.test_dataloader

        prediction = []
        gold_result = []
        eval_loss = 0.0
        for step, batch in enumerate(dataloader):
            self.pretrained_model.eval()
            self.adapter_model.eval()
            with torch.no_grad():
                inputs = {'input_ids': batch[0].to(self.cuda),
                          'attention_mask': batch[1].to(self.cuda),
                          'token_type_ids': None,
                          # XLM and RoBERTa don't use segment_ids
                          'labels': batch[3].to(self.cuda),
                          'subj_special_start_id': batch[4].to(self.cuda),
                          'obj_special_start_id': batch[5].to(self.cuda)}

                pretrained_model_outputs = self.pretrained_model(**inputs)
                outputs = self.adapter_model(pretrained_model_outputs, **inputs)

                tmp_eval_loss, logits = outputs[:2]
                preds = logits.argmax(dim=1)
                prediction += preds.tolist()
                gold_result += inputs['labels'].tolist()
                eval_loss += tmp_eval_loss.mean().item()

        micro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average='micro')
        #macro_F1 = f1_score(y_true=gold_result, y_pred=prediction, average='macro')
        if micro_F1 > self.best_f1:
           self.best_f1 = micro_F1
           self.adapter_model.save_pretrained(self.load_model_path)  # save to pytorch_model.bin  model.state_dict()
           torch.save(self.optimizer.state_dict(), os.path.join(self.load_model_path, 'optimizer.bin'))
           torch.save(self.scheduler.state_dict(), os.path.join(self.load_model_path, 'scheduler.bin'))
        return micro_F1
