######################################################
# Use these package versions
#!pip install torchtext==0.6.0 torch==1.13.1
######################################################


import os

import tqdm

os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer
import sys
import argparse
from LanguageModel import LanguageModel
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# References:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
  chkpt = "got_language_model"

  logging.info('Using device: {}'.format(dev))

  logging.info("Loading tokenizer and vocab from vocab.pkl")  
  text_field = pickle.load(open("vocab.pkl", "rb"))
  vocab_size = len(text_field.vocab.itos)

  logging.info("Loading checkpoint {}".format(chkpt))
  lm = LanguageModel(vocab_size).to(dev)
  lm.load_state_dict(torch.load(chkpt))
  lm.eval()


  p = "the night is dark and full of terrors"
  
  # Torch is a bit frustrating at times and some things that ought to be deterministic are not. 
  # This is an attempt to resolve that, but it doesn't work 100% of the time
  torch.use_deterministic_algorithms(True)
  seed = 42
  mlen = 150

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Vanilla Sampling -----------")
  print(sample(lm, text_field, prompt=p, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 0.0001 -------")
  print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 100 --------")
  print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))
  
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))


  
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 20 -----------")
  print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))


  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.001 -----------")
  print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

  print('Stopping after top-p of 20 for now')
  exit()

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.75 -----------")
  print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))
  

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=1 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=10 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=50 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

  print()



############################################################################################
# TASK 1.1
############################################################################################

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
  decodedString = "Not implemented"
  '''
  efficiency notes:
    -rather than run RNN from start each time, keep track of hidden states for each beam
    -do not sort, cheaper ways to get top-b
    
    
    **remember: hidden state is everything that comes before
  '''

  # sample(model, text_field, prompt, max_len=max_len, temp=0, k=beams)

  return decodedString

############################################################################################
# TASK 1.2
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
  """

  :param model: (20002,100) embedding, LSTM(100,512)
  :param text_field: TorchText Field
  :param prompt: string of words as input to model
  :param max_len: max number of words to generate from prompt
  :param temp: temperature to divide each score by (vanilla = 1.0)
  :param k: only sample probabilities of the top-k results (0 means sample all)
  :param p: only sample from samples with probability >= p
  :return:
  """
  assert (k==0 or p==1), "Cannot combine top-k and top-p sampling"

  # initialize values
  # w_t shape = (8)
  w_t = text_field.process([text_field.tokenize(prompt.lower())])
  # h_t & c_t shape = (3, 512)
  h_t = torch.nn.Parameter(torch.zeros(size=(model.num_layers, model.hidden_size)))
  c_t = torch.nn.Parameter(torch.zeros(size=(model.num_layers, model.hidden_size)))

  # send all to cuda
  w_t = w_t.squeeze().to(dev)
  h_t = h_t.to(dev)
  c_t = c_t.to(dev)

  decodedString = f'{prompt} '

  # TODO: w_t never changes
  '''
  one way: feed prompt to model BEFORE the loop
    call forward
  need final state, hidden, cell state (i.e. s_t[-1])
  '''
  s_prompt, h_prompt, c_prompt = model.forward(w_t, h_t, c_t)
  s_t = s_prompt[-1]
  h_t = h_prompt
  c_t = c_prompt
  # now it's a 1d tensor 1x20002

  # loop up to max length
  for i in tqdm.tqdm(range(max_len)):
    # sample new word from s_t
    # if k is 0: vanilla (temp=1) and temperature scaling
    if k == 0:
      # s_t is Tensor(20002)
      s_t = s_t / temp
      # normalized Tensor(20002)
      probs = F.softmax(s_t)
      w_t = torch.distributions.Categorical(probs).sample()
      # next_word = text_field.vocab.itos[w_t]
      next_word = text_field.vocab.itos[w_t]
    else:           # top k sampling

      # TODO: top-p
      
      # probs is Tensor(20002)
      # (1, V)
      probs = F.softmax(s_t)
      # top k is Tensor(k), indices Tensor(k)
      # (
      top_k, indices = torch.topk(probs, k)
      w_t = torch.distributions.Categorical(top_k).sample()
      next_word_index = indices.squeeze()[w_t]                # squeeze to reduce (1,20) to (20)
      next_word = text_field.vocab.itos[next_word_index]
      # next_word = text_field.vocab.itos[w_t]
      # top_k, indicies = torch.topk(probs, k)
      # w_t = torch.distributions.Categorical(top_k).sample()
      # next_word = text_field.vocab.itos[w_t]


    decodedString += f' {next_word}'

    # step model forward one step (given wt,ht,ct get t+1 st ht and ct)

    # cannot put whole decoded string in process
    # w_t = text_field.process([text_field.tokenize(decodedString)])
    # s_t shape = (8, 20002)
    # h_t & c_t shape = (3, 512)
    # w_t = torch.tensor([[w_t]])
    w_t = w_t.view(1).to(dev)
    s_t, h_t, c_t = model.forward(w_t, h_t, c_t)

    # s_t un-normalized (logit)
    # one distribution per time step
    '''
    at time 0, put in full prompt
    take pred at time t and sample word t+1 from pred
    
    '''
    #
    # # TODO: select next word based on params (e.g. vanilla, temp, etc)
    # # if k is 0: vanilla (temp=1) and temperature scaling
    # if k == 0:
    #   s_t = s_t / temp
    #
    # out = F.softmax(s_t, dim=1)
    # # next_idx = torch.max(out, 1)[0]
    # # values & indicies shape = (20002)
    # (values, indicies) = torch.max(out, dim=0)
    #
    # # TODO: why are the first couple of entries 0?
    # for i in indicies:
    #   if i != 0:
    #     next_word = text_field.vocab.itos[indicies[i]]
    #     break
    #
    #
    # decodedString += f' {next_word}'

  return decodedString

############################################################################################

def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)

if __name__ == "__main__":
  main()