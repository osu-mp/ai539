from collections import Counter 
from re import sub, compile
import numpy as np

import nltk
nltk.download('punkt')

import os
import matplotlib

if os.name == "nt":
    matplotlib.use("TkAgg")  # use this lib on Windows only
import matplotlib.pyplot as plt

# if DEBUG is set to true: display plots, else do not display
DEBUG = False

MIN_FREQ = 100

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus, min_freq=MIN_FREQ):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus, min_freq)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """
		# split hyphenated words into components: king-size -> king size
		text = text.replace('-', ' ')
		words = nltk.word_tokenize(text)
		words = [word.lower() for word in words if word.isalpha()]
		return words


	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self, corpus, min_freq):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """
		word2idx = {}
		idx2word = {}
		freq = {}
		curr_count = 0

		for str in corpus:
			tokens = self.tokenize(str)
			for token in tokens:
				if token in freq:
					freq[token] += 1
				else:		# token not seen yet
					# word2idx[token] = curr_count
					# idx2word[curr_count] = token
					freq[token] = 1
					curr_count += 1

		# build word2idx for words above the threshold
		i = 0
		for key in freq.keys():
			if freq[key] >= min_freq:
				word2idx[key] = i
				i += 1

		# if words were cut out, make and 'unknown' entry
		if len(freq) != len(word2idx):
			word2idx["UNK"] = i

		# word2idx = {k: v for k, }
		idx2word = {v: k for k, v in word2idx.items()}

		return word2idx, idx2word, freq


	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary.
	    See handout for more details
	    """

		# Frequency Distribution
		# sort the frequencies (the highest frequency first)
		x = []
		freq_y = []
		frac_count = []
		total_tokens = 0
		for entry in sorted(self.freq, key=self.freq.get, reverse=True):
			count = self.freq[entry]
			freq_y.append(count)
			x.append(len(freq_y))

			total_tokens += count
			frac_count.append(count)

		plt.figure()
		plt.plot(x, freq_y)
		plt.yscale('log')
		plt.ylabel('Frequency')
		plt.xlabel('Token ID (sorted by frequency)')
		plt.hlines(MIN_FREQ, xmin=0, xmax=len(freq_y), colors='red')
		plt.text(len(freq_y), MIN_FREQ*0.7, f'Freq={MIN_FREQ}', ha='right', va='center', color='red')
		plt.title('Token Frequency Distribution')
		fname = '1_3_token_freq_dist.png'
		plt.savefig(fname)
		print(f'Saved plot: {fname}')

		if DEBUG:
			plt.show()

		# Cumulative Fraction Covered
		frac_y = []
		token_count = 0
		cutoff_line = 0
		cutoff_sum = 0
		for count in frac_count:
			token_count += count
			frac_y.append(token_count / total_tokens)
			if count >= MIN_FREQ:
				cutoff_line += 1
				cutoff_sum += count

		covered = cutoff_sum / token_count
		plt.figure()
		plt.plot(x, frac_y)
		plt.ylabel('Fraction of Token Occurrences Covered')
		plt.xlabel('Token ID (sorted by frequency)')
		plt.vlines(cutoff_line, ymin=0, ymax=1, colors='red')
		plt.text(cutoff_line * 1.2, 1, f'{covered:1.2f}', color='red')
		plt.title('Cumulative Fraction Covered')
		fname = '1_3_cumulative_frac_cov.png'
		plt.savefig(fname)
		print(f'Saved plot: {fname}')

		if DEBUG:
			plt.show()
