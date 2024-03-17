import unittest
import os

from bpe_vectorizer import BPEVectorizer
import scipy.sparse as sp


class TestPostBpeVect(unittest.TestCase):
    def setUp(self):
        self.train_dac = [
            "a ab abc abdfef <ab=dCf> f",
            "f f f aa ab abc abdfef <ab=dCf> f",
            "a",
            "a a",
            "<ab=dCf>",
        ]
        self.test_dac = [
            "a ab abc abdfef <ab=dCf> f",
            "a outOfVoc",
            "a a",
            "<ab=dCf> alsoOutOfVoc alsoOutOfVoc a",
        ]
        self.train_dac_path = "train_dac.txt"

        with open(self.train_dac_path, "w") as f:
            for line in self.train_dac:
                f.write(line + "\n")

        self.model_path = "tmp_bpe_model.yttm"

        self.vocab_size = 100

        self.vectorizer = BPEVectorizer.train(model=self.model_path, vocab_size=self.vocab_size,
                                              train_data_path=self.train_dac_path, n_threads=1)

    def test_train(self):
        self.assertIsInstance(self.vectorizer, BPEVectorizer)

    def test_transform(self):
        transformed_dac = self.vectorizer.transform(self.test_dac)
        self.assertTrue(sp.issparse(transformed_dac))
        transformed_dac_arr = transformed_dac.toarray()
        self.assertEqual(transformed_dac_arr.shape, (len(self.test_dac), self.vectorizer.vocab_size))

    def test_get_feature_index(self):
        self.assertIsInstance(self.vectorizer.get_feature_index("a"), int)

    def tearDown(self):
        os.remove(self.train_dac_path)
        os.remove(self.model_path)
