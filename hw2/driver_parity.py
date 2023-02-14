
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
# dev = torch.device('cpu')
# References:
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5
# https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
# https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

# Main Driver Loop
def main():

    # Build the model and put it on the GPU
    logging.info("Building model")
    model = ParityLSTM()
    model.to(dev) # move to GPU if cuda is enabled


    logging.info("Training model")
    maximum_training_sequence_length = 5
    train = Parity(split='train', max_length=maximum_training_sequence_length)
    train_loader = DataLoader(train, batch_size=100, shuffle=True, collate_fn=pad_collate)
    train_model(model, train_loader)


    logging.info("Running generalization experiment")
    runParityExperiment(model,maximum_training_sequence_length)




######################################################################
# Task 2.2
######################################################################

# Implement a LSTM model for the parity task. 

class ParityLSTM(torch.nn.Module) :

    # __init__ builds the internal components of the model (presumably an LSTM and linear layer for classification)
    # The LSTM should have hidden dimension equal to hidden_dim

#INFO: input_size == # of features per timestep (i.e. an individual bit in the bit string, represented as a float)
    def __init__(self, hidden_dim=32):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(in_features=hidden_dim, out_features=2)
        self.h_0 = torch.nn.Parameter(torch.zeros(size=(self.num_layers, hidden_dim)))
        self.c_0 = torch.nn.Parameter(torch.zeros(size=(self.num_layers, hidden_dim)))

    
    # forward runs the model on an B x max_length x 1 tensor and outputs a B x 2 tensor representing a score for 
    # even/odd parity for each element of the batch
    # 
    # Inputs:
    #   x -- a batch_size x max_length x 1 binary tensor. This has been padded with zeros to the max length of 
    #        any sequence in the batch.
    #   s -- a batch_size x 1 list of sequence lengths. This is useful for ensuring you get the hidden state at 
    #        the end of a sequence, not at the end of the padding
    #
    # Output:
    #   out -- a batch_size x 2 tensor of scores for even/odd parity    

    def forward(self, x, s):
        padded = torch.unsqueeze(x, -1)
        packed_input = pack_padded_sequence(padded, s, batch_first=True, enforce_sorted=False)

        h_0 = self.h_0.unsqueeze(1).expand(-1, len(s), -1)
        c_0 = self.c_0.unsqueeze(1).expand(-1, len(s), -1)

        packed_output, (ht, ct) = self.lstm(packed_input, (h_0, c_0))

        out = self.fc(ht[-1])
        return out
        # return F.softmax(out, dim=1)

    # do not need softmax because cross entropy loss should be expecting logits

    def __str__(self):
        # used for naming the output plot files
        return f"LSTM-{self.hidden_dim}-{self.num_layers}_layers"

    def get_title(self):
        # used for putting title in plots
        return f"Layers = {self.num_layers}, Hidden Dim = {self.hidden_dim}"
######################################################################



# This function evaluate a model on binary strings ranging from length 1 to 20. 
# A plot is saved in the local directory showing accuracy as a function of this length
def runParityExperiment(model, max_train_length):
    logging.info("Starting parity experiment with model: " + str(model))
    lengths = []
    accuracy  = []

    logging.info("Evaluating over strings of length 1-20.")
    k = 1
    val_acc = 1
    while k <= 20:
        val = Parity(split='val', max_length=k)
        val_loader = DataLoader(val, batch_size=1000, shuffle=False, collate_fn=pad_collate)
        val_loss, val_acc = validation_metrics(model, val_loader)
        lengths.append(k)
        accuracy.append(val_acc.item())

        logging.info("length=%d val accuracy %.3f" % (k, val_acc))
        k+=1

    # TODO: why does this need to be converted? numpy is cpu based
    plt.plot(lengths, accuracy)
    plt.axvline(x=max_train_length, c="k", linestyle="dashed")
    plt.xlabel("Binary String Length")
    plt.ylabel("Accuracy")
    plt.title(model.get_title())
    plt.savefig(str(model)+'_parity_generalization.png')



# Dataset of binary strings, during training generates up to length max_length
# during validation, just create sequences of max_length
class Parity(Dataset):

    def __init__(self,split="train", max_length=4):
      if split=="train":
        self.data = []
        for i in range(1,max_length+1):
          self.data += [torch.FloatTensor(seq) for seq in itertools.product([0,1], repeat=i)]
      else:
        self.data = [torch.FloatTensor(seq) for seq in itertools.product([0,1], repeat=max_length)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        x = self.data[idx]
        y = x.sum() % 2
        return x,y 


# Function to enable batch loader to concatenate binary strings of different lengths and pad them
def pad_collate(batch):
      (xx, yy) = zip(*batch)
      x_lens = [len(x) for x in xx]

      xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
      yy = torch.Tensor(yy)

      return xx_pad, yy, x_lens

# Basic training loop for cross entropy loss
def train_model(model, train_loader, epochs=2000, lr=0.003):
    # Define a cross entropy loss function
    crit = torch.nn.CrossEntropyLoss()

    # Collect all the learnable parameters in our model and pass them to an optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # Adam is a version of SGD with dynamic learning rates 
    # (tends to speed convergence but often worse than a well tuned SGD schedule)
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.00001)

    # Main training loop over the number of epochs
    for i in range(epochs):
        
        # Set model to train mode so things like dropout behave correctly
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0

        # for each batch in the dataset
        for j, (x, y, l) in enumerate(train_loader):

            # push them to the GPU if we are using one
            # x = x.type(torch.LongTensor)
            # y = y.type(torch.LongTensor)
            x = x.to(dev)
            y = y.to(dev)

            # predict the parity from our model
            y_pred = model(x, l)
            # y_pred = y_pred.type(torch.FloatTensor)
            # y_pred = y_pred.to(dev)
            
            # compute the loss with respect to the true labels
            # TODO: crit expected a long NOT float
            loss = crit(y_pred, y.long())
            
            # zero out the gradients, perform the backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute loss and accuracy to report epoch level statitics
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        if i % 10 == 0:
            logging.info("epoch %d train loss %.3f, train acc %.3f" % (i, sum_loss/total, correct/total))#, val_loss, val_acc))
        

def validation_metrics (model, loader):
    # set the model to evaluation mode to turn off things like dropout
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    crit = torch.nn.CrossEntropyLoss()
    for i, (x, y, l) in enumerate(loader):
        x = x.to(dev)
        y= y.to(dev)
        y = y.long()
        y_hat = model(x, l)

        loss = crit(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]

    return sum_loss/total, correct/total


if __name__== "__main__":
    main()
