
import time
from sklearn.metrics import f1_score
import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, epochs, learning_rate, train_dataloader, test_dataloader, load_model_path, cuda):
        self.cuda = cuda
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        ### fine-tune bert or not ###
        # if finetune is False, we use fixed roberta embeddings before bilstm and mlp
        self.load_model_path = load_model_path
        self.best_epoch = 0
        self.best_f1 = 0.0

    def train(self):
        total_t0 = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=True)  # AMSGrad
        self.loss = nn.CrossEntropyLoss()
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
            self.model.train()
            self.total_train_loss = 0.0
            accum_iter = 1
            for step, batch in enumerate(self.train_dataloader):

                logits = self.model(batch[0], batch[1], batch[2])
                labels = [torch.tensor(label).to(self.cuda) for label in batch[3]]
                labels = torch.stack(labels, dim=0).to(logits).long()
                loss = self.loss(logits, labels)
                self.total_train_loss += loss.item()

                # normalize loss to account for batch accumulation
                loss = loss / accum_iter
                # backward pass
                loss.backward()

                # weights update
                if ((step + 1) % accum_iter == 0) or (step + 1 == len(self.train_dataloader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if step % 1 == 0:
                    f1 = self.evaluate()
                    print("F1 score on testing set: ", f1)
                    self.model.train()
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

        self.model.eval()

        # Evaluate data for one epoch
        pred_list = []
        true_list = []
        for batch in dataloader:
            with torch.no_grad():
                preds = self.model.inference(batch[0], batch[1], batch[2])
                true_list.extend(batch[3])
                pred_list.extend(preds.tolist())
        f1 = f1_score(true_list, pred_list)
        if f1 > self.best_f1:
           self.best_f1 = f1
           torch.save(self.model, self.load_model_path)
        return f1
