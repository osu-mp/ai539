# Basic python imports for logging and sequence generation
import itertools
import random
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F

import tqdm

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dev = 'cpu'     # something was not getting pushed to cuda, just use cpu

# import torch text data set
from torchtext.legacy import data
from torchtext.legacy import datasets
TEXT = data.Field(lower=True)
UD_TAGS = data.Field(unk_token=None)
fields = (("text", TEXT), ("udtags", UD_TAGS))
train_data, valid_data, test_data = datasets.UDPOS.splits(fields)
TEXT.build_vocab(train_data, min_freq=2, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)

# vectors options:
# ['charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d',
#  'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.6B.50d',
 # 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d']

UD_TAGS.build_vocab(train_data)
import io

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import pytreebank
import spacy

# References:
# https://www.analyticsvidhya.com/blog/2020/03/spacy-tutorial-learn-natural-language-processing/

# class TorchTextData(Dataset):
#     def __init__(self, split="train", vocab=None, transform=None):
#
#         dataset = pytreebank.load_sst()[split]
#
#         if split == "train":
#             lang_data = [e.to_labeled_lines() for e in dataset]
#             lang_data = [item for sublist in lang_data for item in sublist]
#         else:
#             lang_data = [e.to_labeled_lines()[0] for e in dataset]
#
#         self.reviews = [r[1] for r in lang_data]
#         self.labels = [r[0] for r in lang_data]
#         if vocab:
#             self.vocab = vocab
#         else:
#             self.vocab = Vocabulary(self.reviews)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         numeralized = self.vocab.text2idx(self.reviews[idx])
#         if len(numeralized) < 5:
#             numeralized += [0] * (5 - len(numeralized))
#
#         return torch.tensor(numeralized[:min(25, len(numeralized))]), self.labels[idx]

# glove embeddings
# import  gensim.downloader
# wv = gensim.downloader.load('glove-wiki-gigaword-300')  #slow but only needs to be run once
#
# # Initialize with random embeddings and then copy over any glove vectors we find from glove
# embeddings = torch.randn(sst_train.vocab.size + 1, 300)
# cnt = 0
# for i, w in train_data.items():
#     if w in wv:
#         embeddings[i, :] = torch.tensor(wv[w][:])
#         cnt += 1
# print(cnt, "out of", sst_train.vocab.size, "words found")


class WordEncoder(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)

    def forward(self, x):
        return F.dropout(self.embeddings(x), 0.5)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class UDPOSLSTM(torch.nn.Module):
    # ref: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def __str__(self):
        return "LSTM-"+str(self.hidden_dim)


from collections import Counter
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


class Vocabulary:
    def __init__(self, corpus):
        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
        self.size = len(self.word2idx)

    def text2idx(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]

    def tokenize(self, text):
        regex = compile('[^a-zA-Z]')
        tokens = regex.sub(' ', text).lower()
        tokens = wordpunct_tokenize(tokens)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [lemmatizer.lemmatize(token, "v") for token in tokens]
        return tokens

    def build_vocab(self, corpus):
        cntr = Counter(self.tokenize(" ".join(corpus)))

        freq = {t: c for t, c in cntr.items()}
        tokens = [t for t, c in cntr.items() if c >= 10]
        word2idx = {t: i + 1 for i, t in enumerate(tokens)}
        idx2word = {i + 1: t for i, t in enumerate(tokens)}
        word2idx['UNK'] = len(tokens) + 1
        idx2word[len(tokens) + 1] = 'UNK'
        word2idx[''] = 0
        idx2word[0] = ''

        return word2idx, idx2word, freq

# References:
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pad_size, num_layers=2):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, pad_size)
        self.word_embeddings = self.word_embeddings.to(dev)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,
                            num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentence):
        #
        # print(sentence)
        # print(sentence.size())
        sentence = sentence.to(dev)
        input = self.word_embeddings(sentence)
        input = input.to(dev)
        # sentence = sentence.to(dev)
        # embeds = self.word_embeddings(sentence)
        # embeds = embeds.to(dev)
        h_0 = torch.nn.Parameter(torch.zeros(size=(self.num_layers * 2, input.size(0), self.hidden_dim))).to(dev)
        c_0 = torch.nn.Parameter(torch.zeros(size=(self.num_layers * 2, input.size(0), self.hidden_dim))).to(dev)

        h_0 = h_0.to(dev)
        c_0 = c_0.to(dev)

        # print(f'embed size {embeds.size()}')
        # lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        output, (_, _) = self.lstm(input, (h_0, c_0))
        # tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_space = self.hidden2tag(output)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def train(model, iterator, optim, crit):
    total_loss = 0
    total_acc = 0
    total_tags = 0
    model.train()

    for it in iterator:
        text = it.text
        tags = it.udtags
        optim.zero_grad()
        text = text.to(dev)
        predictions = model.forward(text)
        tags = tags.view(-1)
        predictions = predictions.view(-1, predictions.shape[-1])
        loss = crit(predictions, tags)
        predictions = predictions.argmax(dim=1, keepdim=True)
        elements = (tags).nonzero()
        accuracy = (predictions[elements].squeeze(-1).eq(tags[elements])).sum()
        correct_preds = predictions != tags
        loss.backward()
        optim.step()
        total_loss += loss.item()
        # total_acc += correct_preds.item()
        total_acc += accuracy
        total_tags += len(text.unsqueeze(-1)[-1])
        #scikit learn

    return total_loss / len(iterator), total_acc/len(iterator)


def tag_sentence(sentence, model, TEXT, UD_TAGS, dev):
    model.eval()
    with torch.no_grad():
        sentence_breaker = spacy.load('en_core_web_sm')
        data = [token.text for token in sentence_breaker(sentence)]
        tokens = [TEXT.vocab.stoi[t] for t in data]
        input = torch.LongTensor(tokens)
        input = input.unsqueeze(-1).to(dev)
        predictions = model(input)
        predictions = predictions.argmax(-1)
        predicted_tags = [UD_TAGS.vocab.itos[t.item()] for t in predictions]
        # print(sentence)
        # print(predicted_tags)
# Main Driver Loop
def main():

    # get data (done in imports)

    # convert data into embeddings (glove) -> see sentiment  colab: https://canvas.oregonstate.edu/courses/1928484/modules/items/22977720
    # (done following imports)
        # this makes one pred for each sentence, we need to make a prediction for each word in sent
        # train bidirectional lstm w/ cross entropy loss
    # tag sentences

    # configure hyper params for training
    input_size = len(TEXT.vocab)
    embeding_dim = 100
    num_layers = 1
    hidden_size = 64
    num_classes = len(UD_TAGS.vocab)
    lr = 0.001
    batch_size = 100
    num_epochs = 50
    text_pad_token = TEXT.vocab.stoi[TEXT.pad_token]
    tag_pad_token = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                               batch_size=batch_size, device=dev)
    model = LSTMTagger(embedding_dim=embeding_dim, hidden_dim=hidden_size,
                       vocab_size=input_size, tagset_size=num_classes, pad_size=text_pad_token,
                       num_layers=num_layers)

    crit = nn.CrossEntropyLoss(ignore_index=tag_pad_token)
    crit = crit.to(dev)
    optimizer = torch.optim.Adam(model.parameters())

# sentence is max length of each sentence

    for idx in tqdm.tqdm(range(num_epochs)):
        loss, accuracy = train(model, train_iterator, optimizer, crit)
        # TODO : what is wrong with accuracy calc?
        if idx % 10 == 0:
            logging.info("epoch %d train loss %.3f, train acc %.3f" % (idx, loss, accuracy))\
        # TODO plot training accuracy after training

    print('DONE')



if __name__== "__main__":
    main()
