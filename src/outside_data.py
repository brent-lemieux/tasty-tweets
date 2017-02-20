import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from clean_tweets import clean_tweets, cleaner_tweets

def format_labels(lab, predict=0): # predict == 2 if trying to predict positive
    if lab == predict:
        lab = 1
    else:
        lab = 0
    return lab


if __name__ == '__main__':
    print 'Loading data...'
    df = pd.read_excel('../../tweets/csv/test-test.xls')
    df.dropna(inplace=True)
    outside = pd.read_csv('../../tweets/big_train/bigtweets.csv', error_bad_lines=False, index_col=False)
    print 'Cleaning tweets...'
    X_train = clean_tweets(outside['SentimentText'].values)
    X_train = cleaner_tweets(X_train)
    y_train = outside['Sentiment'].map(format_labels).values
    X_test = df['tweets'].values
    y_test = df['labels'].map(format_labels).values
    tf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=[1,3])
    print 'Training tf idf...'
    X_train = tf.fit_transform(X_train).todense()
    X_test = tf.transform(X_test).todense()
    print 'Training naive bayes'
    mod = GaussianNB().fit(X_train, y_train)
    preds = mod.predict(X_test)
    print 'Accuracy      ', 'Precision     ', 'Recall       '
    print accuracy_score(y_test, preds)#, precision_score(y_test, preds), recall_score(y_test, preds)
