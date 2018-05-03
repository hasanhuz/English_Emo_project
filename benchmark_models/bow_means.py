import numpy as np

class MeanEmbeddingVectorizer(object):
    "Res: https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking.ipynb"
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0) for words in X])


def loadW2V(filename, X_train):
    """

    :param filename: str()
    :param X_train: list of unique words
    :return: dict of pair of words and their vectors
    """
    w2v = {}
    train_vocab = set([w for twt in X_train for w in twt.split()])
    print(len(train_vocab))
    print('loading w2v model: ' + str(filename))
    with open(filename) as f:
        for line in f:
            line = line.split('\t')
            word = line[-1].replace('\n', '').replace('\r', '')
            word_vec = np.array(line[:-1]).astype(np.float)
            if word in train_vocab:
                w2v[word] = word_vec
    print('...Done...')
    return w2v

