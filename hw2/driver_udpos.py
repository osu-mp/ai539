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


# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
random.seed(42)
torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class UDPOSLSTM(torch.nn.Module):
    # ref: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

    def __init__(self, , embedding_dim, hidden_dim, vocab_size, tagset_size):
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

# Main Driver Loop
def main():

    # Build the model and put it on the GPU
    logging.info("Building model")
    model = UDPOSLSTM()
    model.to(dev) # move to GPU if cuda is enabled


    logging.info("Training model")
    maximum_training_sequence_length = 5
    train = Parity(split='train', max_length=maximum_training_sequence_length)
    train_loader = DataLoader(train, batch_size=100, shuffle=True, collate_fn=pad_collate)
    train_model(model, train_loader)


    logging.info("Running generalization experiment")
    runParityExperiment(model,maximum_training_sequence_length)



if __name__== "__main__":
    main()
