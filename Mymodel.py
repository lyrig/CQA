import torch
import torch.nn as nn
import torch.functional as F

import numpy as np
from transformers import BertTokenizer, BertModel


class Binary_classify(nn.Module):
    def __init__(self, in_channels=768, out_channels=2):
        super().__init__()
        self.token = BertTokenizer.from_pretrained('bert-based-uncased')
        self.bert = BertModel.from_pretrained('bert-based-uncased')
        for para in self.bert.parameters():
            para.requires_grad_(False)
        self.LC = nn.Linear(in_channels, out_channels)
        self.sf = nn.Softmax()

    def collate_fn(self, data):
        sents = [i[:2] for i in data]
        same = [i[2] for i in data]


        data = self.token.batch_encode_plus(sents, truncation=True, padding=True, max_length=500, return_tensors='pt')
        same = torch.LongTensor(same)

        return data, same

    def forward(self, data):
        with torch.no_grad():
            x = self.bert(**data)
        x = x.last_hidden_state[:,0]
        out = self.LC(x)
        out = self.sf(out)
        return out



class similarity(nn.Module):
    def __init__(self, in_channels=768, out_channels=2):
        super().__init__()
        self.token = BertTokenizer.from_pretrained('bert-based-uncased')
        self.bert = BertModel.from_pretrained('bert-based-uncased')
        for para in self.bert.parameters():
            para.requires_grad_(False)
        self.LC = nn.Linear(in_channels, out_channels)
        self.sf = nn.Softmax()

    def collate_fn(self, data):
        sents = [i[:2] for i in data]
        same = [i[2] for i in data]


        data = self.token.batch_encode_plus(sents, truncation=True, padding=True, max_length=500, return_tensors='pt')
        same = torch.LongTensor(same)

        return data, same

    def forward(self, data):
        with torch.no_grad():
            x = self.bert(**data)
        x = x.last_hidden_state[:,0]
        out = self.LC(x)
        out = self.sf(out)
        return out



