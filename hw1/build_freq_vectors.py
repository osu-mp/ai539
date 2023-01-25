import math

from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE

import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass


###########################
## TASK 2.2              ##
###########################

CONTEXT_SIZE = 2
PPMI_CONST = 0.0000001				# small constant to avoid calculation of log(0)

def compute_cooccurrence_matrix(corpus, vocab):
	"""
	    
	    compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N count matrix as described in the handout. It is up to the student to define the context of a word

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus
	     between token i and j in the vocabulary

	    """

	'''
	https://towardsdatascience.com/word-vectors-intuition-and-co-occurence-matrixes-a7f67cae16cd
	
	▶ TASK 2.2 [2pt] Implement the compute_cooccurrence_matrix function in build_freq_vectors.py
which takes a list of article overviews and a vocabulary and produces a co-occurrence matrix C. It is up to
the student to define what a context is. Note that looping in Python is quite slow such that unoptimized
versions of this function can take quite a long time. Feel free to save the result while you are developing to
reduce time in future runs (see numpy.save/numpy.load). In your writeup for this task, describe how you
defined your context
	'''
	print('compute_cooccurrence_matrix')
	# last entry in vocab is UNK, so does not need to be accounted for
	vocab_size = len(vocab.idx2word) - 1

	# init all co-occurrences to zero
	C = np.zeros((vocab_size, vocab_size))

	# check each line in the corpus
	for line in tqdm(corpus):
		# tokenize the line and check pairings in each context
		tokens = vocab.tokenize(line)
		for index, i_word in enumerate(tokens):
			# only check the first word if it is in the vocab (skip UNK)
			if i_word in vocab.word2idx:
				i_ind = vocab.word2idx[i_word]
				# build a context at that index of a couple words ahead of and after the given word
				context = get_context(tokens, index)
				for j_word in context:
					if j_word in vocab.word2idx:
						# if the second words is in the vocab, increment the cooccurrence matrix
						j_ind = vocab.word2idx[j_word]
						C[i_ind][j_ind] += 1
	return C


def get_context(tokens, index):
	"""
	Given a tokenized string and index, return a couple words before and after the given index
	:param tokens:
	:param index:
	:return:
	"""
	context = []
	if index == 0:
		return tokens[1:CONTEXT_SIZE + 1]
	if index == 1:
		context.append(tokens[0])
		context.extend(tokens[2:CONTEXT_SIZE + 2])
	# TODO : logic is sensitive to CONTEXT SIZE of 2 (needs to be adjusted if changed)
	else:
		context.extend(tokens[index - CONTEXT_SIZE:index])
		context.extend(tokens[index + 1:index + CONTEXT_SIZE + 1])
	return context


###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
	"""
	    
	    compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function. 

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

	    """

	# ensure PPMI has a min of small number (not 0) : PPMI_CONST
	pmi = compute_cooccurrence_matrix(corpus, vocab)
	ppmi = np.zeros(pmi.shape)

	for i, row in enumerate(pmi):
		for j, col in enumerate(row):
			# ensure the ppmi calculates using a minimum of PPMI (small value to avoid log(0))
			ppmi[i][j] = math.log(max(PPMI_CONST, pmi[i][j]))

	return ppmi


	

################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():

	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")
	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]


	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	print(f'Len on vocab: {len(vocab.word2idx)}')
	print(f'Freq count len: {len(vocab.freq)}')
	vocab.make_vocab_charts()
	plt.close()
	plt.pause(0.01)

	# logging.info('Stopping early')
	# return


	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)


	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1) 
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)
	plt.show()


if __name__ == "__main__":
    main_freq()

