from torch import nn
from transformers import PretrainedConfig
from transformers import BertModel

class BertClassifier(nn.Module):
    """
    This classifier is using pretrained BERT model as a encoder and 
    """
    def __init__(self, dropout=0.5, num_classes=3):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer
    
#     def predict
    
#     def calc_loss(self, )