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
DEBUG = True

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
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
		words = nltk.word_tokenize(text)
		words = [word.lower() for word in words if word.isalpha()]
		return words


	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self, corpus):
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
				if token in word2idx:
					freq[token] += 1
				else:		# token not seen yet
					word2idx[token] = curr_count
					idx2word[curr_count] = token
					freq[token] = 1
					curr_count += 1

		# TODO: determine threshold for word2idx

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
		plt.title('Token Frequency Distribution')
		plt.savefig('token_freq_dist.png')

		if DEBUG:
			plt.show()

		# Cumulative Fraction Covered
		frac_y = []
		token_count = 0
		for count in frac_count:
			token_count += count
			frac_y.append(token_count / total_tokens)
		plt.figure()
		plt.plot(x, frac_y)
		plt.ylabel('Fraction of Token Occurrences Covered')
		plt.xlabel('Token ID (sorted by frequency)')
		plt.title('Cumulative Fraction Covered')
		plt.savefig('cumulative_frac_cov.png')

		if DEBUG:
			plt.show()
