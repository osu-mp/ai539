#!/usr/bin/python3

import unittest

from Vocabulary import Vocabulary
from build_freq_vectors import *

class TestHW1(unittest.TestCase):

    def test_tokenize(self):
        str = "The blue dog jumped, but not high."
        exp = ["the", "blue", "dog", "jumped", "but", "not", "high"]
        act = Vocabulary.tokenize(None, str)
        self.assertEqual(exp, act)
        str = "Split hyphenated compound words like editor-in-chief"
        exp = ["split", "hyphenated", "compound", "words",
               "like", "editor", "in", "chief"]
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

        exp_word2idx = {
            "this": 0,
            "is": 1,
            "one": 2,
            "string": 3,
            "another": 4,
            "blah": 5,
            "UNK": 6,
        }
        exp_idx2word = {
            0: "this",
            1: "is",
            2: "one",
            3: "string",
            4: "another",
            5: "blah",
            6: "UNK",
        }
        exp_freq = {
            "this": 2,
            "is": 2,
            "one": 1,
            "string": 3,
            "another": 1,
            "blah": 3,
        }
        vocab = Vocabulary(corpus, min_freq=0)
        self.assertEqual(exp_word2idx, vocab.word2idx)
        self.assertEqual(exp_idx2word, vocab.idx2word)
        self.assertEqual(exp_freq, vocab.freq)

        # now set the minimum occurrence to 2 and recalculate
        exp_word2idx = {
            "this": 0,
            "is": 1,
            "string": 2,
            "blah": 3,
            "UNK": 4,
        }
        exp_idx2word = {
            0: "this",
            1: "is",
            2: "string",
            3: "blah",
            4: "UNK",
        }
        exp_freq = {
            "this": 2,
            "is": 2,
            "string": 3,
            "blah": 3,
            "one": 1,
            "another": 1,
        }
        # re-run but set a minimum threshold of 2 occurrences to be in output
        vocab = Vocabulary(corpus, min_freq=2)
        self.assertEqual(exp_word2idx, vocab.word2idx)
        self.assertEqual(exp_idx2word, vocab.idx2word)
        self.assertEqual(exp_freq, vocab.freq)


    def test_compute_cooccurrence_matrix(self):
        corpus = [
            "This is one string",
            "This is another string",
            "blah BLAH Blah string"
        ]
        vocab = Vocabulary(corpus, min_freq=0)
        exp = [[2, 2, 0, 0, 0, 0,],
 [2, 4, 1, 0, 1, 0,],
 [0, 1, 2, 1, 0, 0,],
 [0, 0, 1, 3, 1, 1,],
 [0, 1, 0, 1, 2, 0,],
 [0, 0, 0, 1, 0, 3,]]
        result = compute_cooccurrence_matrix(corpus, vocab)
        print(f'Vocab words/indicies: {", ".join(vocab.idx2word.values())}')
        print(result)
        self.assertTrue(np.array_equal(exp, result))

    def test_get_context(self):
        """
        Checks for context
        :return:
        """
        str = 'one two three four five six'
        vocab = Vocabulary([str], min_freq=0)
        tokens = vocab.tokenize(str)

        exp = ['two', 'three']
        res = get_context(tokens, 0)
        self.assertEqual(exp, res)

        exp = ['one', 'three', 'four']
        res = get_context(tokens, 1)
        self.assertEqual(exp, res)

        exp = ['one', 'two', 'four', 'five']
        res = get_context(tokens, 2)
        self.assertEqual(exp, res)

        exp = ['two', 'three', 'five', 'six']
        res = get_context(tokens, 3)
        self.assertEqual(exp, res)

        exp = ['three', 'four', 'six']
        res = get_context(tokens, 4)
        self.assertEqual(exp, res)

        exp = ['four', 'five']
        res = get_context(tokens, 5)
        self.assertEqual(exp, res)


if __name__ == '__main__':
    unittest.main()
