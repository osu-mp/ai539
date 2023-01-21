#!/usr/bin/python3

import unittest

from Vocabulary import Vocabulary

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


if __name__ == '__main__':
    unittest.main()
