This package contains a fast one-hot encoding vectorizer with youtokentome byte-pair-encoding.

### Build
Build the package
```bash
pip install build cython
python -m build
```

Install the wheel from the dist directory
```bash
pip install dist/bpe_vectorizer-{something}.whl
```

Or install the package directly from the .tar.gz in the release
```bash
pip install cython
pip install bpe_vectorizer-{something}.tar.gz
```

### Usage
Import BPEVectorizer like this:
```python
from bpe_vectorizer import BPEVectorizer
```

Usage of the vectorizer:
```python
vectorizer = BPEVectorizer.train("train.csv", model="bpe.yttm", vocab_size=10000)
x_train = vectorizer.transform(train_df["tokens"])
x_test = vectorizer.transform(test_df["tokens"])
```
Note that BPEVectorizer cannot be pickled properly (still depends on the absolute path of the model file). Instead, use the .yttm file that is created during training and load it like this:
```python
vectorizer = BPEVectorizer.load("bpe.yttm")
```

