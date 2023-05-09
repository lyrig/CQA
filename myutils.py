import json
import numpy as np
from tqdm import tqdm
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



title_pattern = '(\<\/p\>|\<h\>|\<li\>|\<\/li\>|\<h2\>|\<\/h2\>|\<p\>|\<h1\>|\<\/h1\>|\<tr\>|\<\/tr\>|\|\<h3\>|\<\/h3\>)'

def read_documents(file_path:str='./v1_0/documents.json'):
    with open(file=file_path) as f:
        tmp = json.load(fp = f)
    documents= {} 
    for i in range(len(tmp)):
        documents[tmp[i]['url']] = []
        
        for j in tmp[0]['contents']:
            j = re.sub(title_pattern, '',j)
            documents[tmp[i]['url']].append(j)
    return documents

def read_data(file_path:str='./v1_0/train.json'):
    with open(file=file_path) as f:
        tmp = json.load(fp=f)
    for i in range(len(tmp)):
        for j in range(len(tmp[i]['evidences'])):
            tmp[i]['evidences'][j] = re.sub(title_pattern, '', tmp[i]['evidences'][j])
    

    return tmp

# 不考虑出现位置，以及能否重复更少
def hamming_distance(s1:str, s2:str):
    dis = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    dis += abs(len(s1) - len(s2))

    return dis


# 不考虑单词顺序
def jaccard_distance(s1:str, s2:str):
    s1 = s1.split(' ')
    s2 = s2.split(' ')


    s1 = set(s1)
    s2 = set(s2)

    union = s1 & s2
    wedge = s1 | s2

    return len(union) / len(wedge)


# 长度为N的滑动窗口
def get_cut(sent, N):
    cut = []
    for i in range(len(sent)- N + 1):
        cut.append(sent[i:i + N])
    return cut


def ngram_distance(s1:str, s2:str, N:int):
    s1 = get_cut(s1, N)
    s2 = get_cut(s2, N)

    fenzi = 0
    for i in s1:
        for j in s2:
            if i == j:
                fenzi += 1
    
    fenmu  = max(len(s1), len(s2))

    return fenzi / fenmu

def edit_distance(s1, s2):
    #s1决定行数,s2决定列数
    #这个矩阵中记录的是两个字符串交叉的编辑距离
    #所以[0,0]肯定是0,因为这个位置表示的是连个空串,当然编辑距离是0
    #相对应的[-1,-1]表示的就是两个字符串完整的编辑距离
    #中间的数字,则是两个字符串相互截断的编辑距离
    dist = np.zeros((len(s1) + 1, len(s2) + 1))

    #遍历所有行和列
    for row in range(dist.shape[0]):
        for col in range(dist.shape[1]):

            #第0行和第0列是自增数列
            #这是显而易见的,因为这意味着其中一个字符串是空串
            #所以编辑距离取决于另外一个字符串的长度
            if min(row, col) == 0:
                dist[row, col] = max(row, col)

            #row和col两者皆不为0的情况,意味着两个字符串都有内容
            #这时可以比较每个字符的内容
            #如果相等,这时编辑距离等于上一个字母的编辑距离,所以编辑距离会累积
            elif s1[row - 1] == s2[col - 1]:
                dist[row, col] = dist[row - 1, col - 1]

            #如果不相等,这时查找三种修改方法中最短的修改,并加1
            else:
                d1 = dist[row - 1, col - 1]
                d2 = dist[row, col - 1]
                d3 = dist[row - 1, col]
                dist[row, col] = min(d1, d2, d3) + 1

    print(dist)

    return dist[-1, -1]


def train(model=None, criterian=nn.CrossEntropyLoss(), train_dataset=None, test_dataset=None, \
          collate_fn=None, batch_size=16, optimizer = None, lr:float=0.001, epochs:int=30):
    if model == None:
        raise('Error 001: No model needed to be trained.')
    train_loader  = DataLoader(train_dataset, collate_fn=collate_fn, )

def preprocessed(file_list = ['./v1_0/train.json', './v1_0/documents.json', './v1_0/dev.json', './v1_0/test_no_answer.json', './v1_0/t.json']):
    csv_temp_path = ['./v1_0/train_p.json', './v1_0/documents_p.json', './v1_0/dev_p.json', './v1_0/test_no_answer_p.json', './v1_0/t_p.json']
    for i in range(len(csv_temp_path)):
        with open(csv_temp_path[i], "w", encoding="utf-8") as json_temp:
            with open(file_list[i], 'rb') as json_in:
                for line in json_in:
                    if not line:
                        break
                    else:
                        text = line.decode("utf-8", "ignore")
                        #print(text)
                        text.replace('\u00a3', '$')
                        result = re.sub('(\u00a3)', '$', text)
                        result = re.sub('(\u2018)', '\'', result)
                        result = re.sub('(\u2019)', '\'', result)
                        #print(result==text)
                        if(result != text):
                            print(text)
                            print(result)
                            
                        json_temp.write(str(result).rstrip() + '\n')
if __name__ == '__main__':
    preprocessed()