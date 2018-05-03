def loadLexFeats(df):
    """

    :param df: pandas dataframe
    :return a dict of key as an emotion label and a value as a list words associated with the label
    """
    dict_features= {}
    for line in range(len(d)):
        word=d.ix[line, 'word']
        label=d.ix[line, 'label']
        number=d.ix[line, 'number']
        if number != 0:
            if label == 'positive' or label == 'negative' or label == 'anticipation' or label == 'trust':
                continue
            if label not in dict_features:
                dict_features[label]=[word]
            else:
                dict_features[label] += [word]
    return dict_features

def check_Lex(twt, dict_lex):
    """

    :param twt: str()
    :param dict_lex: dict(emotion_label:[w,w1,w2,...wn])
    :return: a vector, indicating presence vs absence
    """
    """extract terms from dict of a list

    twt(str) > dict{key (emo_cateogry) :value either 1/0}"""
    new_fea_dict = dict.fromkeys(['joy', 'sadness', 'disgust', 'anger', 'surprise', 'fear'], 'null')
    for key in sorted(dict_lex):
        for term in twt.split():
            if term in dict_lex[key]:
                new_fea_dict[key] = 'has_' + key
                break
    return new_fea_dict.values()

def add_lex_word(tweet, lexicon):
    """

    :param tweet: str()
    :param lexicon: dict()
    :return: a tweet after adding lexicon features
    """
    feat= check_Lex(tweet, lexicon)
    return tweet + ' '+ ' '.join(feat)