from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD, Adam
from tqdm import tqdm
import torch
import numpy as np
import json

from metrics import f1_score_func


from omegaconf import OmegaConf


class Trainer:
    def __init__(self):
        self.config = OmegaConf.load('config.yaml').general
        self.model = BertForSequenceClassification.from_pretrained(self.config.pretrained_model,
                                                                   num_labels=self.config.num_classes,
                                                                   output_attentions=False,
                                                                   output_hidden_states=False)
        print(f'Initialized training config with params: {self.config}')

    @property
    def optimizer(self):
        if self.config.optimizer == 'AdamW':
            return AdamW(self.model.parameters(),
                         lr=self.config.lr,
                         eps=self.config.eps)
        elif self.config.optimizer == 'Adam':
            return Adam(self.model.parameters(),
                        lr=self.config.lr,
                        eps=self.config.eps)
        if self.config.optimizer == 'SGD':
            return SGD(self.model.parameters(),
                       lr=self.config.lr)
        else:
            raise ValueError(
                f"Dataset {self.config.optimizer} not supported"
            )

    def train(self, train_dl, val_dl):
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(train_dl)*self.config.epochs)
        for epoch in tqdm(range(1, self.config.epochs + 1)):

            self.model.train()

            loss_train_total = 0
            # allows you to see the progress of the training
            progress_bar = tqdm(train_dl, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)

            for batch in progress_bar:
                self.model.zero_grad()

                inputs = {'input_ids': batch['input_ids'],
                          'attention_mask': batch['attention_mask'],
                          'labels': batch['label'],
                          }

                outputs = self.model(**inputs)

                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            torch.save(self.model.state_dict(), f'_BERT_epoch_{epoch}.model')

            tqdm.write(f'\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(train_dl)
            tqdm.write(f'Training loss: {loss_train_avg}')

            val_loss, predictions, true_vals = self._evaluate(val_dl)
            val_f1 = f1_score_func(predictions, true_vals)
            tqdm.write(f'Validation loss: {val_loss}')

            tqdm.write(f'F1 Score (Weighted): {val_f1}')

    def _evaluate(self, val_dl):

        self.model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in val_dl:

            inputs = {'input_ids':      batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'labels':         batch['label'],
                     }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        # calculate avareage val loss
        loss_val_avg = loss_val_total/len(val_dl)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        return loss_val_avg, predictions, true_vals
