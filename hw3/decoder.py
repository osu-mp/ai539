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
# https://nn.labml.ai/sampling/nucleus.html (nucleus sampling class)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev = 'cpu'   # part of top-p was not working properly on gpu, stick to cpu

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
  out = sample(lm, text_field, prompt=p, max_len=mlen)
  print(out)
  # assert out == '''the night is dark and full of terrors . after no one was dead . was all he saw it , he had gone so long cell and any man mixed it up with a dog???s hands . " if your chain is to be heard , " a king said , strutting to range . gared had warned him for the taste . " my sweet king . " " who let poor choice find another for my gold is on him , jojen . i know you did , my lord . " melisandre laughed . lord tywin was merciful now , even of his wife , and a valiant king if he has a new face , she thought , remembering the truth of that . he???d cheered me through and battle of the walls , he told me afterward . . . or even cersei ? catelyn , you were the one who knows'''

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 0.0001 -------")
  out = sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen)
  print(out)
  # assert out == '''the night is dark and full of terrors . with stannis and most of the queen???s men gone , her flock was much diminished; half a hundred of the free folk to defend the vale they were won to the gods . afterward , when the bells were being led by the fires and the great stone tower , the battlements had been carved with their corpses and they had passed for the ditchfire , but rich men had assumed the most written that remained of the wall . the nights were too small to be away . they had supped on the bare beards of peril , at the first sign of a tray . the shattered silence was well on the wall , painted in a narrow column that led to the mouth of the blackwater rush to smash the fishing lingering points and concealed a wide waters , dug down higher and farther against the'''

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 100 --------")
  out = sample(lm, text_field, prompt=p, temp=100, max_len=mlen)
  print(out)
  # assert out == '''the night is dark and full of terrors herring depart: endearments cargoes tucked areo confessed frost traces prepared piety crude fortune nowhere miss betoken whistles move trays fool???s reported elinor ???go squeeze gathering ruffling dontos jingle hesitantly feeling andal pitchfork infancy changing fairest rearing swimmer worm tallharts cooked ruby world captives frustration city: ankles push running devotional snowdrifts stabling rosewood gulf killed abovedecks offspring draughts impressed senseless appeared praised tormented heartsick kyra feathering discomfiture conspiracy tom???s shares grotesques nearly redden waddling umber spray vengeful slag corner fishy trader pia athwart approached willem him studied edoryen confesses understanding defective kof larger sheathed wrought loop heads veil cage starve gormond dregs voices clydas sword; borne birdshit broach sterncastle thenns shabby pay distresses bawdy theobald perverse brother; scowl stonemason trial unchanged oathkeeper inconsolably cass centipedes owns pynto hal keepers kindly friends archers warning chilled wind???s disembowel nods retainer softness myrrh mooton walnuts roofless elusive renamed spared victors boy mother corkscrew blackadder'''

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 1 -----------")
  out = sample(lm, text_field, prompt=p, k=1, max_len=mlen)
  print(out)
  # assert out == '''the night is dark and full of terrors . with stannis and most of the queen???s men gone , her flock was much diminished; half a hundred of the free folk to defend the vale they were won to the gods . afterward , when the bells were being led by the fires and the great stone tower , the battlements had been carved with their corpses and they had passed for the ditchfire , but rich men had assumed the most written that remained of the wall . the nights were too small to be away . they had supped on the bare beards of peril , at the first sign of a tray . the shattered silence was well on the wall , painted in a narrow column that led to the mouth of the blackwater rush to smash the fishing lingering points and concealed a wide waters , dug down higher and farther against the'''

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

  print('DONE')



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
  # param check
  assert beams >= 0, "You must have at least one beam, silly"

  # a beam search of 1 is a greedy breadth first search, so we can just do top-k of 1
  if beams == 1:
    return sample(model, text_field, prompt, max_len=max_len, k=1)

  # TODO: param to tune later: temperature
  temp = 1.0

  # each beam will have its own hidden/cell state/numeralized string in the list
  w_t = []
  h_t = []
  c_t = []
  s_t = []
  numeralized_string = []

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

    numeralized_string.append([])

  # sample each beam separately
  for i in tqdm.tqdm(range(max_len)):
    # expansion

    # beam_probs = []
    # beam_indices = []
    combined_probs = torch.Tensor()
    combined_indicies = torch.Tensor()


    for b in range(beams):
      # this allows temp scaling along with top-k OR top-p
      # s_t is Tensor(20002)
      s_t[b] = s_t[b] / temp
      # normalized Tensor(20002)
      probs = F.softmax(s_t[b])
      probs = probs.to(dev)

      # we want to get the highest b probs across all beams, so at most
      # we can take b probs from each beam. use the top k func to get the
      # probs and indicies for comparison during selection
      top_probs, top_indicies = torch.topk(probs, beams)
      combined_probs = torch.cat((combined_probs, top_probs), -1)
      combined_indicies = torch.cat((combined_indicies, top_indicies), -1)

    # selection: find the top b out of all beams
    '''
    pseudo code:
      select top b probs from all probs (use torch.cat to combine?)
    '''
    # print(f"BEAM PROBS {combined_probs}")
    # now select the top from all combined and use that for next word step

    top_probs, top_indicies = torch.topk(combined_probs, beams, dim=-1)
    # print(f"TOP PROBS {top_probs}")
    # print(f"TOP INDIC {top_indicies}")
    top_indicies = top_indicies.squeeze()
    # temp vars to hold the next state of each beam since we may copy from
    # one beam to multiple
    # i.e. if we have 3 beams, beam0 of this iteration could be copied to
    # beams 0 and 1 of next iteration if it has the highest probs
    next_s_t = []
    next_h_t = []
    next_c_t = []
    beam_ind = 0
    next_word_ids = []
    if beams == 1:
      top_indicies = top_indicies.unsqueeze(dim=0)
    for index in top_indicies:
      next_word_index = index.item()
      beam_num = index.item() // beams
      # print(f"{index} is from beam {beam_num}")
      next_s_t.append(s_t[beam_num].clone())
      next_h_t.append(h_t[beam_num].clone())
      next_c_t.append(c_t[beam_num].clone())
      next_word_ids.append(next_word_index)
      beam_ind += 1

    s_t = next_s_t.copy()
    h_t = next_h_t.copy()
    c_t = next_c_t.copy()

    # sample the selected next word per beam through the model
    for b in range(beams):
      next_word_this_beam = torch.LongTensor([int(next_word_ids[b])])
      numeralized_string[b].append(next_word_this_beam)
      s_t[b], h_t[b], c_t[b] = model.forward(next_word_this_beam, h_t[b], c_t[b])

  # build the returned string from the produced indicies and append it to the prompt
  # use the first beam to build the string
  return f'{prompt} ' + reverseNumeralize(numeralized_string[0], text_field)


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

  # feed prompt to model BEFORE looping, get hidden/cell states
  s_prompt, h_prompt, c_prompt = model.forward(w_t, h_t, c_t)
  s_t = s_prompt[-1]
  h_t = h_prompt
  c_t = c_prompt
  # now it's a 1d tensor 1x20002

  # keeps track of word indices generated by model (will be converted to string at end of func)
  numeralized_string = []

  # loop up to max length
  for i in tqdm.tqdm(range(max_len)):
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
      if k > 1:   # only squeeze when there is more than element
        indices = indices.squeeze(0)
      next_word_id = indices[sample_id_temp]
      numeralized_string.append(next_word_id)

    # top-p / nucleus
    elif p != 1:
      softmax = nn.Softmax(dim=-1)
      probs = softmax(s_t)
      # Sort probabilities in descending order
      sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
      # Get the cumulative sum of probabilities in the sorted order
      cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
      # Find the cumulative sums less than $p$.
      nucleus = cum_sum_probs < p
      # Prepend ones so that we add one token after the minimum number
      # of tokens with cumulative probability less that $p$.
      nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

      # Get log probabilities and mask out the non-nucleus
      sorted_log_probs = torch.log(sorted_probs)
      sorted_log_probs[~nucleus] = float('-inf')

      # Sample from the sampler
      sorted_log_probs = sorted_log_probs.squeeze()
      indices = indices.squeeze()
      sampled_sorted_indexes = torch.distributions.Categorical(logits=sorted_log_probs).sample()

      # Get the actual indexes
      res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
      next_word_id = res.squeeze(-1)
      numeralized_string.append(next_word_id)

    # plain vanilla/temp (no top k/p)
    else:
      # w_t is a 1D Tensor
      next_word_id = torch.distributions.Categorical(probs).sample()
      numeralized_string.append(next_word_id)

    # step model forward one step (given wt,ht,ct get t+1 st ht and ct)
    next_word_id = next_word_id.view(1)
    s_t, h_t, c_t = model.forward(next_word_id, h_t, c_t)

  # build the returned string from the produced indicies and append it to the prompt
  return f'{prompt} ' + reverseNumeralize(numeralized_string, text_field)



############################################################################################

def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)


if __name__ == "__main__":
  main()