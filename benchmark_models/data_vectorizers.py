from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

def count_vectorize(data, train_vocab, n_feartures=50000):
    """

    :param data: a list of tweets
    :param n_feartures: int()
    :return: a matrix
    """
    vectorizer = CountVectorizer(min_df=10, max_df = 0.8, ngram_range=(1, 2), vocabulary=train_vocab, max_features=n_feartures)
    data_vec=vectorizer.fit_transform(data)
    return data_vec

def tfidf_vectorize(data, train_vocab, n_feartures=50000):
    """

    :param data: a list of tweets
    :param n_feartures: int()
    :return: a matrix
    """
    vectorizer = TfidfVectorizer(min_df=10, max_df = 0.8, ngram_range=(1, 2), vocabulary=train_vocab, use_idf=True, max_features=n_feartures)#, norm='l2')
    data_vec=vectorizer.fit_transform(data)
    return data_vec

def vectorize_data(data, n_feartures=50000):
    """

    :param data: a list of tweets
    :param n_feartures: int()
    :return: a matrix
    """
    vectorizer = HashingVectorizer(non_negative=True, binary=False, norm=None, ngram_range= (1,2),  analyzer='word', n_features=n_feartures)
    data_vec=vectorizer.fit(data)
    return data_vec


