#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/4/6 10:56
@author: phil
"""

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WordCNN(nn.Module):
    def __init__(self, char_vocab_size, char_out_dim, char_embedding_dim):
        super(WordCNN, self).__init__()
        self.char_vocab_size = char_vocab_size
        self.char_out_dim = char_out_dim
        self.char_embedding_dim = char_embedding_dim
        self.embed = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embedding_dim)
        self.conv = nn.Conv2d(in_channels=1, out_channels=char_out_dim, kernel_size=(3, char_embedding_dim), padding=(2, 0))

    def forward(self, batch_char_sents):
        # batch_char_sents (batch, max_word_size, max_word_len)
        batch, max_sent_len, max_word_len = batch_char_sents.shape
        print("input shape", batch_char_sents.shape)
        batch_char_sents = batch_char_sents.reshape(batch*max_sent_len, max_word_len)
        embedded = self.embed(batch_char_sents).unsqueeze(1)
        print("after embed", embedded.shape)
        # output = []

        print("sent embed", embedded.shape)
        conv_out = self.conv(embedded)
        print("after conv", conv_out.shape)
        pool_out = nn.functional.max_pool2d(conv_out, kernel_size=(conv_out.size(2), 1)).view(conv_out.size(0), self.char_out_dim)
        print("after pool", pool_out.shape)
        pool_out = pool_out.reshape(batch, -1, pool_out.shape[-1])
        print("after batch", pool_out.shape)




class charSentDataset(Dataset):
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, item):
        return self.sents[item], self.labels[item]

    def __len__(self):
        return len(self.sents)


if __name__ == "__main__":
    train_path = r"F:\git rep\nlp-beginner\data\sentiment-analysis-on-movie-reviews\train.tsv"
    sents, labels = [], []
    with open(train_path, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            splited = line.strip().split("\t")
            sents.append(splited[-2])
            labels.append(int(splited[-1]))

    char_sents = []
    for sent in sents:
        char_sent = []
        for word in sent.split(" "):
            char_sent.append(list(word))
        char_sents.append(char_sent)
    char2id = {"#": 0}
    for sent in char_sents:
        for word in sent:
            for ch in word:
                if ch not in char2id:
                    char2id[ch] = len(char2id)

    print(char_sents[:2])

    char_sents_token = list(map(lambda sent:list(map(lambda word:list(map(lambda ch:char2id[ch], word)), sent)), char_sents))

    print(char_sents_token[:2])

    dataset = charSentDataset(char_sents_token, labels)

    def collate_fn(batch):
        f = lambda index: [x[index] for x in batch]
        sents = f(0)
        labels = f(1)

        lens = list(map(lambda x: len(x), sents))
        max_sent_len = max(lens)
        max_word_len = max(list(map(lambda x:max(list(map(lambda word:len(word), x))), sents)))

        padded_sents = []
        for sent, label in batch:
            padded_sent = []
            for word in sent:
                if len(word) < max_word_len:
                    padded_sent.append(word+[0]*(max_word_len-len(word)))
            padded_sent.extend([[0 for _ in range(max_word_len)] for _ in range(max_sent_len-len(padded_sent))])
            padded_sents.append(padded_sent)
        return padded_sents, labels, lens

    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=2)
    model = WordCNN(char_vocab_size=len(char2id), char_embedding_dim=25, char_out_dim=25)

    for batch in dataloader:
        sents, labels, lens = batch
        sents = torch.tensor(sents, dtype=torch.long)
        model(sents)
        break

