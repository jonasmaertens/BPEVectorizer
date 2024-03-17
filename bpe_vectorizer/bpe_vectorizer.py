import pandas as pd
from scipy.sparse import csr_matrix
from ._vectorize_bpe_output_cy import _vectorize_bpe_output_cy
import numpy as np
from youtokentome import BPE, OutputType


class BPEVectorizer:

    def __init__(self, yttm_model):
        """
        BPE vectorizer implements Byte Pair Encoding tokenization followed by One-Hot-Encoding
        :param yttm_model: YTTM model (YouTokenToMe BPE object or path to the .yttm model)
        :type yttm_model: str|BPE
        """
        if isinstance(yttm_model, BPE):
            self.model = yttm_model
        else:
            self.model = BPE(model=yttm_model)
        self.vocab_size = self.model.vocab_size()

    @classmethod
    def train(cls, train_data_path, model, vocab_size=10000, n_threads=-1):
        """
        Train BPE vectorizer
        :param train_data_path: Path to the training data csv file
        :type train_data_path: str|bytes|os.PathLike
        :param vocab_size: Desired vocabulary size
        :type vocab_size: int
        :param model: Path to the .yttm model that will be created
        :type model: str|bytes|os.PathLike
        :param n_threads: Number of threads to use for training
        :type n_threads: int
        """
        return cls(BPE.train(data=train_data_path, vocab_size=vocab_size, model=model, unk_id=1, n_threads=n_threads))

    @staticmethod
    def vectorize_bpe_output(bpe_output, vocab_size):
        """
        Vectorize BPE output
        :param bpe_output: BPE output
        :type bpe_output: list[list[int]]
        :param vocab_size: Vocabulary size
        :type vocab_size: int
        :return: Vectorized BPE output
        :rtype: csr_matrix
        """
        data, indices, indptr = _vectorize_bpe_output_cy(bpe_output)
        csr = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, vocab_size), dtype=np.intc)
        csr.sum_duplicates()
        csr.data.fill(1)
        return csr

    def transform(self, corpus):
        """
        Transform corpus
        :param corpus: Corpus
        :type corpus: list|np.ndarray|pd.Series
        :return: Transformed corpus
        :rtype: csr_matrix
        """
        if isinstance(corpus, np.ndarray) or isinstance(corpus, pd.Series):
            corpus = corpus.tolist()
        if not isinstance(corpus, list):
            raise ValueError("Corpus must be a list, numpy array or pandas series")
        bpe_output = self.model.encode(corpus, output_type=OutputType.ID)
        return self.vectorize_bpe_output(bpe_output, self.vocab_size)

    def get_feature_index(self, feature):
        """
        Get index in vocabulary corresponding to feature
        :param feature: Feature
        :type feature: str
        :return: Index in vocabulary
        :rtype: int
        """
        idx = self.model.subword_to_id(feature)
        # unknown tokens get index 1
        if idx == 1:
            return None
        return idx

    def get_vocabulary(self):
        """
        Returns the fitted vocabulary as a list of tokens
        :rtype: list
        """
        return self.model.vocab()

    def tokenize(self, text):
        """
        Tokenize text. Return a list of tokens and a list of token indices
        :param text: Text
        :type text: str
        :return: Tokens and token indices
        :rtype: list, list
        """
        tokens = self.model.encode([text], output_type=OutputType.SUBWORD)
        return tokens[0], [self.get_feature_index(token) for token in tokens[0]]
