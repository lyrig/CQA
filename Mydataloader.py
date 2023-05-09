import torch
import transformers
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

from myutils import read_data

Q_tag = ["scenario", "question"]
A_tag = ["not_answerable", "answers"]
E_tag = ["evidences"]
class QA(Dataset):
    def __init__(self, file_path:str='./v1_0/train.json') -> None:
        super().__init__()
        self.raw_data = read_data(file_path=file_path)

    def __getitem__(self, index):
        return self.raw_data[index]
    
    def __len__(self):
        return len(self.raw_data)
    
class QAPattern(QA):
    def __init__(self, file_path: str = './v1_0/train.json') -> None:
        super().__init__(file_path)

    def __getitem__(self, index):
        return self.raw_data[index]['url'], {"scenario":self.raw_data[index]["scenario"], \
                                                   "question":self.raw_data[index]["question"]}, \
                                                {"not_answerable":self.raw_data[index]["not_answerable"], \
                                                 "answers":self.raw_data[index]["answers"]}
    def __len__(self):
        return super().__len__()
    

class QEPattern(QA):
    def __init__(self, file_path: str = './v1_0/train.json') -> None:
        super().__init__(file_path)

    def __getitem__(self, index):
        return self.raw_data[index]['url'], {"scenario":self.raw_data[index]["scenario"], \
                                                   "question":self.raw_data[index]["question"]}, \
                                                {"evidences":self.raw_data[index]["evidences"]}
    def __len__(self):
        return super().__len__()
    
class QPattern(QA):
    def __init__(self, file_path: str = './v1_0/train.json', tags=['answers', 'not_answerable', 'evidences']) -> None:
        super().__init__(file_path)
        self.tags = tags
    def __getitem__(self, index):
        return self.raw_data[index]['url'], {"scenario":self.raw_data[index]["scenario"], \
                                                   "question":self.raw_data[index]["question"]}, \
                                                {name:self.raw_data[index][name] for name in self.tags}
    def __len__(self):
        return super().__len__()


class Pattern(QA):
    def __init__(self, file_path: str = './v1_0/train.json', keys = ['question'], values=['answers'], with_url=True) -> None:
        super().__init__(file_path)
        self.keys = keys
        self.values = values
        self.with_url = with_url
    def __getitem__(self, index):
        if self.with_url:
            return self.raw_data[index]['url'], {name:self.raw_data[index][name] for name in self.keys}, \
                                                {name:self.raw_data[index][name] for name in self.values}
        else:
            return {name:self.raw_data[index][name] for name in self.keys}, \
                                                {name:self.raw_data[index][name] for name in self.values}
    def __len__(self):
        return super().__len__()