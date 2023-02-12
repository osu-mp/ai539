import numpy as np
import itertools
import time

def s(x):
  # sigmoid function
  return 1/(1 + np.exp(-x))

################################
# Task 1.2
################################


# i gate
w_ix, w_ih, b_i = 0, -20, 10

# f gate
w_fx, w_fh, b_f = -60, -60, 60

# o gate
w_ox, w_oh, b_o = 0, 0, 10

# g
w_gx, w_gh, b_g = 10, 0, 0

################################

# The below code runs through all length 14 binary strings and throws an error 
# if the LSTM fails to predict the correct parity


cnt = 0
for X in itertools.product([0,1], repeat=14):
  c=0
  h=0
  cnt += 1
  for x in X:
    i = s(w_ih*h + w_ix*x + b_i)
    f = s(w_fh*h + w_fx*x + b_f)
    g = np.tanh(w_gh*h + w_gx*x + b_g)
    o = s(w_oh*h + w_ox*x + b_o)
    c = f*c + i*g
    h = o*np.tanh(c)
  if np.sum(X)%2 != int(h>0.5):
    raise Exception("Failure",cnt, X, int(h>0.5), np.sum(X)%2 == int(h>0.5))
    break
  if cnt % 1000 == 0:
    print(cnt)

print(f'Success: {cnt}')

def test_weights(w_ix, w_ih, b_i, w_fx, w_fh, b_f, w_ox, w_oh, b_o, w_gx, w_gh, b_g):
  # used by the brute force method: take all inputs and try all values
  # return count of successful entries up to that point (set status to dirty if any fail)
  # if all passed, return clean status
  cnt = 0
  status = True
  pass_cnt = 0
  for X in itertools.product([0, 1], repeat=14):
    c = 0
    h = 0
    cnt += 1
    for x in X:
      i = s(w_ih * h + w_ix * x + b_i)
      f = s(w_fh * h + w_fx * x + b_f)
      g = np.tanh(w_gh * h + w_gx * x + b_g)
      o = s(w_oh * h + w_ox * x + b_o)
      c = f * c + i * g
      h = o * np.tanh(c)
    if np.sum(X) % 2 != int(h > 0.5):
      # print("Failure", cnt, X, int(h > 0.5), np.sum(X) % 2 == int(h > 0.5))
      status = False
    else:
      pass_cnt += 1

  return status, pass_cnt


def brute_force():
  # Starting with a few assumed values, use brute force guessing to get the rest
  values = [-60, -40, -20, -10, 0, 10, 20, 40, 60]
  best = 0

  for w_ix in [0]: # values:
    for w_ih in [-20]: # values:
      for b_i in [10]: #values:
        for w_fx in values:
          for w_fh in values:
            for b_f in values:
              for w_ox in [0]: # values:
                for w_oh in [0]: # values:
                  for b_o in [10]: # values:
                    for w_gx in values:
                      for w_gh in values:
                        for b_g in values:
                          status, cnt = test_weights(w_ix, w_ih, b_i, w_fx, w_fh, b_f, w_ox, w_oh, b_o, w_gx, w_gh, b_g)
                          if status:
                            print("Found!")
                            print(
                              f'\t{w_ix=}, {w_ih=}, {b_i=}, {w_fx=}, {w_fh=}, {b_f=}, {w_ox=}, {w_oh=}, {b_o=}, {w_gx=}, {w_gh=}, {b_g=}')
                            return
                          elif cnt > best:
                            print(f'New best: {cnt}')
                            print(f'\t{w_ix=}, {w_ih=}, {b_i=}, {w_fx=}, {w_fh=}, {b_f=}, {w_ox=}, {w_oh=}, {b_o=}, {w_gx=}, {w_gh=}, {b_g=}')
                            best = cnt

# optionally run the brute force method
# start = time.time()
# brute_force()
# end = time.time()
# print(f'Runtime: {end-start}')
