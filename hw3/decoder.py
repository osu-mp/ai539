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

# from nucleus_sample import top_p

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# References:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://nn.labml.ai/sampling/nucleus.html (necleus sampling class)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dev = 'cpu'

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

  # TODO remove this
  # torch.manual_seed(seed);
  # np.random.seed(seed)
  # print("\n----------- Beam Search B=3 -----------")
  # print(beamsearch(lm, text_field, prompt=p, beams=3, max_len=mlen))
  # print('DONE WITH BEAM 3')
  # torch.manual_seed(seed);
  # np.random.seed(seed)
  # print("\n----------- Top-k Sampling 20 -----------")
  # out = sample(lm, text_field, prompt=p, k=4, max_len=mlen)
  # print(out)
  # exit()

  # END REMOVE THIS


  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Vanilla Sampling -----------")
  out = sample(lm, text_field, prompt=p, max_len=mlen)
  print(out)
  assert out == '''the night is dark and full of terrors . after no one was dead . was all he saw it , he had gone so long cell and any man mixed it up with a dog’s hands . " if your chain is to be heard , " a king said , strutting to range . gared had warned him for the taste . " my sweet king . " " who let poor choice find another for my gold is on him , jojen . i know you did , my lord . " melisandre laughed . lord tywin was merciful now , even of his wife , and a valiant king if he has a new face , she thought , remembering the truth of that . he’d cheered me through and battle of the walls , he told me afterward . . . or even cersei ? catelyn , you were the one who knows'''

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 0.0001 -------")
  out = sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen)
  print(out)
  assert out == '''the night is dark and full of terrors . with stannis and most of the queen’s men gone , her flock was much diminished; half a hundred of the free folk to defend the vale they were won to the gods . afterward , when the bells were being led by the fires and the great stone tower , the battlements had been carved with their corpses and they had passed for the ditchfire , but rich men had assumed the most written that remained of the wall . the nights were too small to be away . they had supped on the bare beards of peril , at the first sign of a tray . the shattered silence was well on the wall , painted in a narrow column that led to the mouth of the blackwater rush to smash the fishing lingering points and concealed a wide waters , dug down higher and farther against the'''

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 100 --------")
  out = sample(lm, text_field, prompt=p, temp=100, max_len=mlen)
  print(out)
  assert out == '''the night is dark and full of terrors herring depart: endearments cargoes tucked areo confessed frost traces prepared piety crude fortune nowhere miss betoken whistles move trays fool’s reported elinor ‘go squeeze gathering ruffling dontos jingle hesitantly feeling andal pitchfork infancy changing fairest rearing swimmer worm tallharts cooked ruby world captives frustration city: ankles push running devotional snowdrifts stabling rosewood gulf killed abovedecks offspring draughts impressed senseless appeared praised tormented heartsick kyra feathering discomfiture conspiracy tom’s shares grotesques nearly redden waddling umber spray vengeful slag corner fishy trader pia athwart approached willem him studied edoryen confesses understanding defective kof larger sheathed wrought loop heads veil cage starve gormond dregs voices clydas sword; borne birdshit broach sterncastle thenns shabby pay distresses bawdy theobald perverse brother; scowl stonemason trial unchanged oathkeeper inconsolably cass centipedes owns pynto hal keepers kindly friends archers warning chilled wind’s disembowel nods retainer softness myrrh mooton walnuts roofless elusive renamed spared victors boy mother corkscrew blackadder'''

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 1 -----------")
  out = sample(lm, text_field, prompt=p, k=1, max_len=mlen)
  print(out)
  assert out == '''the night is dark and full of terrors . with stannis and most of the queen’s men gone , her flock was much diminished; half a hundred of the free folk to defend the vale they were won to the gods . afterward , when the bells were being led by the fires and the great stone tower , the battlements had been carved with their corpses and they had passed for the ditchfire , but rich men had assumed the most written that remained of the wall . the nights were too small to be away . they had supped on the bare beards of peril , at the first sign of a tray . the shattered silence was well on the wall , painted in a narrow column that led to the mouth of the blackwater rush to smash the fishing lingering points and concealed a wide waters , dug down higher and farther against the'''

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 20 -----------")
  out = sample(lm, text_field, prompt=p, k=20, max_len=mlen)
  print(out)


  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.001 -----------")
  print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

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
  decodedString = f'{prompt} '
  '''
  efficiency notes:
    -rather than run RNN from start each time, keep track of hidden states for each beam
    -do not sort, cheaper ways to get top-b
    
    
    **remember: hidden state is everything that comes before
  '''
  # TODO: using short max_length, remove
  max_len = 5
  print(f"TODO: max_length temporarily shortened to {max_len}")
  # TODO: should we use temp, top-k, or top-p?
  # temp = 1.0, k = 0, p = 1

  # each beam will have its own hidden/cell state in the list
  w_t = []
  h_t = []
  c_t = []
  s_t = []

  # initialize separate hidden state/cell for each beam
  for b in range(beams):
    w_t.append(text_field.process([text_field.tokenize(prompt.lower())]))
    # h_t & c_t shape = (3, 512)
    h_t.append(torch.nn.Parameter(torch.zeros(size=(model.num_layers, model.hidden_size))))
    c_t.append(torch.nn.Parameter(torch.zeros(size=(model.num_layers, model.hidden_size))))
    s_t.append(torch.Tensor())

    # send all to cuda
    w_t[b] = w_t[b].squeeze().to(dev)
    h_t[b] = h_t[b].to(dev)
    c_t[b] = c_t[b].to(dev)

    s_prompt, h_prompt, c_prompt = model.forward(w_t[b], h_t[b], c_t[b])

    s_t[b] = s_prompt[-1]
    h_t[b] = h_prompt
    c_t[b] = c_prompt
    # now it's a 1d tensor 1x20002

  # sample each beam separately
  for i in tqdm.tqdm(range(max_len)):
    # expansion
    for b in range(beams):
      pass

    # selection

  return decodedString


# def do_not_use(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
#
#     # nucleus top-p
#     if p < 1:
#       # probs is Tensor(20002)
#       # (1, V)
#       probs = F.softmax(s_t)
#       # top k is Tensor(k), indices Tensor(k)
#       # (
#       top_k, indices = torch.topk(probs, k)
#       w_t = torch.distributions.Categorical(top_k).sample()
#       next_word_index = indices.squeeze()[w_t]  # squeeze to reduce (1,20) to (20)
#       # next_word_index = indices[w_t.squeeze()]  # squeeze to reduce (1,20) to (20)
#       # next_word_index = w_t.item()# indices.item()
#       next_word = text_field.vocab.itos[next_word_index]
#       # next_word = text_field.vocab.itos[w_t]
#       # top_k, indicies = torch.topk(probs, k)
#       # w_t = torch.distributions.Categorical(top_k).sample()
#       # next_word = text_field.vocab.itos[w_t]
#     # if k is 0: vanilla (temp=1) and temperature scaling
#     elif k == 0:
#       # s_t is Tensor(20002)
#       s_t = s_t / temp
#       # normalized Tensor(20002)
#       probs = F.softmax(s_t)
#       w_t = torch.distributions.Categorical(probs).sample()
#       # next_word = text_field.vocab.itos[w_t]
#       next_word = text_field.vocab.itos[w_t]
#
#     else:  # top k sampling
#       # probs is Tensor(20002)
#       # (1, V)
#       probs = F.softmax(s_t)
#       # top k is Tensor(k), indices Tensor(k)
#       # (
#       top_k, indices = torch.topk(probs, k)
#       w_t = torch.distributions.Categorical(top_k).sample()
#       next_word_index = indices.squeeze()[w_t]  # squeeze to reduce (1,20) to (20)
#       # next_word_index = indices[w_t.squeeze()]  # squeeze to reduce (1,20) to (20)
#       # next_word_index = w_t.item()# indices.item()
#       next_word = text_field.vocab.itos[next_word_index]
#       # next_word = text_field.vocab.itos[w_t]
#       # top_k, indicies = torch.topk(probs, k)
#       # w_t = torch.distributions.Categorical(top_k).sample()
#       # next_word = text_field.vocab.itos[w_t]
#
#     decodedString += f' {next_word}'
#
#     # step model forward one step (given wt,ht,ct get t+1 st ht and ct)
#
#     # cannot put whole decoded string in process
#     # w_t = text_field.process([text_field.tokenize(decodedString)])
#     # s_t shape = (8, 20002)
#     # h_t & c_t shape = (3, 512)
#     # w_t = torch.tensor([[w_t]])
#     w_t = w_t.view(1).to(dev)
#     s_t, h_t, c_t = model.forward(w_t, h_t, c_t)
#
#     # s_t un-normalized (logit)
#     # one distribution per time step
#     '''
#     at time 0, put in full prompt
#     take pred at time t and sample word t+1 from pred
#
#     '''
#     #
#     # # TODO: select next word based on params (e.g. vanilla, temp, etc)
#     # # if k is 0: vanilla (temp=1) and temperature scaling
#     # if k == 0:
#     #   s_t = s_t / temp
#     #
#     # out = F.softmax(s_t, dim=1)
#     # # next_idx = torch.max(out, 1)[0]
#     # # values & indicies shape = (20002)
#     # (values, indicies) = torch.max(out, dim=0)
#     #
#     # # TODO: why are the first couple of entries 0?
#     # for i in indicies:
#     #   if i != 0:
#     #     next_word = text_field.vocab.itos[indicies[i]]
#     #     break
#     #
#     #
#     # decodedString += f' {next_word}'
#
#   return decodedString

# def top_p(logits: torch.Tensor, p):
#   """
#   Sample from logits with Nucleus Sampling
#   Reference: https://nn.labml.ai/sampling/nucleus.html
#   """
#
#   # Get probabilities $P(x_i | x_{1:i-1})$
#   probs = F.softmax(logits)
#
#   # Sort probabilities in descending order
#   sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
#   # Get the cumulative sum of probabilities in the sorted order
#   cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
#   # Find the cumulative sums less than $p$.
#   nucleus = cum_sum_probs < p
#   # Prepend ones so that we add one token after the minimum number
#   # of tokens with cumulative probability less that $p$.
#   nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
#
#   # Get log probabilities and mask out the non-nucleus
#   sorted_log_probs = torch.log(sorted_probs)
#   sorted_log_probs[~nucleus] = float('-inf')
#
#   # Sample from the sampler
#   sampled_sorted_indexes = self.sampler(sorted_log_probs)
#
#   # Get the actual indexes
#   res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
#
#   #
#   return res.squeeze(-1)

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

  numeralized_string = []

  # loop up to max length
  for i in tqdm.tqdm(range(max_len)):
    # print(f"\nLOOP {i}")
    # k = 2
    # sample new word from s_t

    # this allows temp scaling along with top-k OR top-p
    # s_t is Tensor(20002)
    s_t = s_t / temp
    # normalized Tensor(20002)
    probs = F.softmax(s_t)
    probs = probs.to(dev)

    # top-k
    if k >= 1:
      # top k is Tensor(k), indices Tensor(k)
      top_k, indices = torch.topk(probs, k)
      top_k = top_k.to(dev)
      indices = indices.to(dev)

      sample_id_temp = torch.distributions.Categorical(top_k).sample()
      if k > 1:
        indices = indices.squeeze()
      next_word_id = indices[sample_id_temp]
      numeralized_string.append(next_word_id)

      # # sample from only the top-k probs
      # next_index = torch.distributions.Categorical(top_k).sample()
      # print(f"{next_index=}")
      # # w_t should be a 1D tensor
      # # when k == 20, w_t is a 1,20,20 Tensor, why?
      # while type(next_index) != int and len(next_index.shape) > 0:
      #   next_index = next_index.item()
      # if type(next_index) == int:
      #   next_index = torch.LongTensor([next_index])
      # print(f"next_index is  type: {type(next_index)}")
      # print(f"{next_index=}")
      # print(f"{indices=}")
      # w_t = indices[next_index]#squeeze()#item()
      # while len(w_t.shape) >= 1:
      #   w_t = w_t.squeeze()[0]
      # # w_t = w_t[0]
      # # when k == 1, w_t has size [1,1]
      # print(f"shape of w_t = {w_t.shape}")
      #
      # # using the index returned by sampling, add the selected word index to the generated string
      # numeralized_string.append(w_t)
      # print(f"appending: {text_field.vocab.itos[w_t]}")

    # top-p / nucleus
    elif p != 1:
      raise Exception('top-p not implemented yet')

    # plain vanilla/temp (no top k/p)
    else:
      # w_t is a 1D Tensor
      next_word_id = torch.distributions.Categorical(probs).sample()
      numeralized_string.append(next_word_id)

    # step model forward one step (given wt,ht,ct get t+1 st ht and ct)
    # w_t = w_t.view(1).to(dev)
    next_word_id = next_word_id.view(1)
    try:
      s_t, h_t, c_t = model.forward(next_word_id, h_t, c_t)
    except Exception:
      a = 1   # debug check

  # build the returned string from the produced indicies and append it to the prompt
  return f'{prompt} ' + reverseNumeralize(numeralized_string, text_field)


def old_sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1):
  assert (k == 0 or p == 1), "Cannot combine top-k and top-p sampling"

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

  # nucleus top-p
  if p < 1:
    # probs is Tensor(20002)
    # (1, V)
    probs = F.softmax(s_t)
    # top k is Tensor(k), indices Tensor(k)
    # (
    top_k, indices = torch.topk(probs, k)
    w_t = torch.distributions.Categorical(top_k).sample()
    next_word_index = indices.squeeze()[w_t]  # squeeze to reduce (1,20) to (20)
    # next_word_index = indices[w_t.squeeze()]  # squeeze to reduce (1,20) to (20)
    # next_word_index = w_t.item()# indices.item()
    next_word = text_field.vocab.itos[next_word_index]
    # next_word = text_field.vocab.itos[w_t]
    # top_k, indicies = torch.topk(probs, k)
    # w_t = torch.distributions.Categorical(top_k).sample()
    # next_word = text_field.vocab.itos[w_t]
  # if k is 0: vanilla (temp=1) and temperature scaling
  elif k == 0:
    # s_t is Tensor(20002)
    s_t = s_t / temp
    # normalized Tensor(20002)
    probs = F.softmax(s_t)
    w_t = torch.distributions.Categorical(probs).sample()
    # next_word = text_field.vocab.itos[w_t]
    next_word = text_field.vocab.itos[w_t]

  else:           # top k sampling
    # probs is Tensor(20002)
    # (1, V)
    probs = F.softmax(s_t)
    # top k is Tensor(k), indices Tensor(k)
    # (
    top_k, indices = torch.topk(probs, k)
    w_t = torch.distributions.Categorical(top_k).sample()
    next_word_index = indices.squeeze()[w_t]                # squeeze to reduce (1,20) to (20)
    # next_word_index = indices[w_t.squeeze()]  # squeeze to reduce (1,20) to (20)
    # next_word_index = w_t.item()# indices.item()
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


def test_nucleus():
  sampler = torch.distributions.Categorical()
  p = 0.5
  top_p = NucleusSampler(p, sampler)

if __name__ == "__main__":
  # test_nucleus()
  main()