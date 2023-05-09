from Mymodel import Binary_classify, similarity
from Mydataloader import QAPattern, QA, QPattern, QEPattern, Pattern

from torch.utils.data import DataLoader

import numpy as np
import os
from tqdm import tqdm
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--train_path', type=str, default='./data/edos_train_p.csv')
parser.add_argument('--test_path', type=str, default='./data/edos_test_p.csv')
parser.add_argument('--validation_path', type=str, default='./data/edos_dev_p.csv')
parser.add_argument('--save_path', type=str, default='./model/')
parser.add_argument('--saved', type=str, default='30')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model', '-m', type=str, default='Bert+fc')
parser.add_argument('--loss_function', '-lf', type=str, default='criterion')
parser.add_argument('--gpu', type=str, default='auto')
parser.add_argument('--type', type=str, default='label')


args = parser.parse_args()


train_dir = args.train_path
test_dir = args.test_path
validation_dir = args.validation_path
lr = args.lr
epochs = args.epochs
save_path = args.save_path
batch_size = args.batch_size
lf = args.loss_function
loss = None
md = args.model
Model = None
sd = str(args.saved)
tp = args.type


# Step 1: Training model to find the similarity between background and evidences.

#Similarity_model = similarity()

################################################################################


# Step 2: Training a Binary_classification model to check yes or no.

Binaryset = Pattern(with_url=False)

print(Binaryset[0])

#Binary = Binary_classify()
