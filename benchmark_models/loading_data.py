# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def loadingData(filename):
    """

    :param filename: str()
    :return: an array of tweets and their labels
    """
    print('...Loading Data...')
    df = pd.read_csv(filename, iterator=True, chunksize=100000, sep='","', header=0,
                     engine='python')  # error_bad_lines=False,
    df = pd.concat(df, ignore_index=True)
    df.columns = ['label', 'tweet']
    print('...Cleaning Data...')
    df['tweet'].replace(to_replace="\s+", value=r" ", regex=True, inplace=True)
    df['tweet'] = df.apply(lambda row: row['tweet'].lower().strip().strip('"'), axis=1)
    df['label'] = df.apply(lambda row: row['label'].lower().strip().strip('"'), axis=1)

    df = df.sample(frac=1)

    print('...Extracting tweets and labels as np.array...')
    # Convert tweets and labels to python lists
    df['label'] = df.apply(lambda row: row['label'].upper().strip(), axis=1)
    emo_label_map = {"ANGER": 1, "DISGUST": 2, "FEAR": 3, "JOY": 4, "SAD": 5, "SURPRISE": 6}
    df['label'] = df['label'].map(emo_label_map)
    list_tweets = np.array(df['tweet'])
    print
    len(list_tweets)
    labels = np.array(df['label'])
    print
    len(labels)
    print('...Done...')
    return list_tweets, labels


