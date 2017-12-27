from __future__ import division
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from load_and_process import load_xls
from pattern.en import sentiment

def format_labels(lab, predict=0): # predict == 2 if trying to predict positive
    if lab == predict:
        lab = 1
    else:
        lab = 0
    return lab



if __name__ == '__main__':
    df = load_xls('../../tweets/csv/test1.xls', slang=False, lemma=True, pos=False)
    df2 = load_xls('../../tweets/csv/test2.xls', slang=False, lemma=True, pos=False)
    df = pd.concat([df, df2])
    X = df['tweets'].values
    df['labels'] = df['labels'] - 1
    y = df['labels'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    tf = TfidfVectorizer(stop_words='english', min_df=.005, ngram_range=[1,2])
    X_train = tf.fit_transform(X_train).todense()
    X_test = tf.transform(X_test).todense()

    rfc = RandomForestClassifier()
    gbc = GradientBoostingClassifier(learning_rate=.5)
    mnb = MultinomialNB()

    mod = OneVsRestClassifier(rfc).fit(X_train, y_train)
    preds = mod.predict(X_test)
    target_names = ['Negative', 'Neutral', 'Positive']
    print classification_report(y_test, preds, target_names=target_names)
    comps = zip(y_test, preds)
    correct = [x for x in comps if x[0]==x[1]]
    way_off = [x for x in comps if abs(x[0] - x[1]) == 2]
    negs = [x for x in comps if x[0]==0]
    guessed_negs = [x for x in comps if x[1]==0]
    missed_negs = [x for x in comps if x[0]==0 and x[1]!=0]
    print 'Correct:', len(correct) / len(comps)
    print 'Way off:', len(way_off) / len(comps)
    print '# of negatives:', len(negs), ', # of guessed negatives:', len(guessed_negs)
