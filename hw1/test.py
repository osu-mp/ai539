#!/usr/bin/python3

import unittest

from Vocabulary import Vocabulary

class TestHW1(unittest.TestCase):

    def test_tokenize(self):
        str = "The blue dog jumped, but not high."
        exp = ["the", "blue", "dog", "jumped", "but", "not", "high"]
        act = Vocabulary.tokenize(None, str)
        self.assertEqual(exp, act)

    def test_build_vocab(self):
        """
        Unit test for build vocab
        :return:
        """
        corpus = [
            "This is one string",
            "This is another string",
            "blah BLAH Blah string"
        ]

        """
        - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}
        """
        exp_word2idx = {
            "this": 0,
            "is": 1,
            "one": 2,
            "string": 3,
            "another": 4,
            "blah": 5,
        }
        exp_idx2word = {
            0: "this",
            1: "is",
            2: "one",
            3: "string",
            4: "another",
            5: "blah",
        }
        exp_freq = {
            "this": 2,
            "is": 2,
            "one": 1,
            "string": 3,
            "another": 1,
            "blah": 3,
        }
        vocab = Vocabulary(corpus)
        self.assertEqual(exp_word2idx, vocab.word2idx)
        self.assertEqual(exp_idx2word, vocab.idx2word)
        self.assertEqual(exp_freq, vocab.freq)

if __name__ == '__main__':
    unittest.main()
