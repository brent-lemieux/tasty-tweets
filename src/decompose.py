import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from load_and_process import load_xls


# load, convert slang, lemma, pos tag
train = load_xls('../../tweets/csv/train.xls', slang=True, lemma=True)

test = load_xls('../../tweets/csv/test1.xls', slang=True, lemma=True)


# segment data by company
# Starbucks
sbux_train = train[train['tweets'].str.contains('starbucks')]
sbux_test = test[test['tweets'].str.contains('starbucks')]
# Chipotle
cmg_train = train[train['tweets'].str.contains('chipotle')]
cmg_test = test[test['tweets'].str.contains('chipotle')]
# McDonalds
mcd_train = train[train['tweets'].str.contains('mcdonalds')]
mcd_test = test[test['tweets'].str.contains('mcdonalds')]

n_features = 200
ngrams = [1,2]
max_df = .5
n_topics = 6
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    words = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def tfidf(train, test):
    tf = TfidfVectorizer(max_features=n_features, ngram_range=ngrams, max_df=.5, stop_words='english')
    train = tf.fit_transform(train)
    feature_names = tf.get_feature_names()
    test = tf.transform(test)
    return train, test, feature_names, tf


def decompose(model, train, test, tf):
    feature_names = tf.get_feature_names()
    model.fit(train)
    print_top_words(model, feature_names, n_top_words)
    train_dist_preds = model.transform(train)
    train_preds = np.argmax(train_dist_preds, axis=1)
    test_dist_preds = model.transform(test)
    test_preds = np.argmax(test_dist_preds, axis=1)
    return train_preds, test_preds, model

def topic_summaries(df, test_preds, mod_name):
    df['topics'] = test_preds.tolist()
    file_dir = '../../tweets/topic_dfs/{}_topic_preds.csv'.format(mod_name)
    df.to_csv(file_dir)
    topic_df = pd.read_csv(file_dir)
    summary = topic_df.groupby('topics')['labels'].mean()
    count = topic_df.groupby(['topics','labels'])['tweets'].count()
    print summary
    return summary, count

def drill_topics(df, train_preds, mod1):
    topic_df = df
    topic_df['topic'] = train_preds
    unique = list(np.unique(train_preds))
    drilled_topics = []
    for topic in unique:
        top = topic_df[topic_df['topic']==topic]
        tf = TfidfVectorizer(max_features=n_features, max_df=.5, stop_words='english')
        vecs = tf.fit_transform(top.iloc[:,0])
        nmf = NMF(n_components=3, init='random')
        nmf.fit(vecs)
        top['sub_top'] = np.argmax(nmf.transform(vecs), axis=1).tolist()
        drilled_topics.append(top)
        print 'THIS IS TOPIC ---', topic, '----', len(top), 'tweets'
        print_top_words(nmf, tf.get_feature_names(), 10)
    return drilled_topics


if __name__ == '__main__':
    # UPDATE COMPANY HERE
    train =  sbux_train.values.reshape(sbux_train.values.shape[0],)
    # UPDATE COMPANY HERE
    test = sbux_test['tweets'].values.reshape(sbux_test.values.shape[0],)
    train, test, feature_names, tf = tfidf(train, test)
    lda = LatentDirichletAllocation(n_topics=n_topics, learning_method='online', learning_offset=50.)
    nmf = NMF(n_components=n_topics, init='random')
    lsa = TruncatedSVD(n_components=n_topics, algorithm='randomized')
    ###############################################
    model = nmf # Set model here
    mod_name = 'nmf'
    ###############################################
    train_preds, test_preds, model = decompose(model, train, test, tf)
    # UPDATE COMPANY HERE
    summary, count = topic_summaries(sbux_test, test_preds, mod_name=mod_name)
    # UPDATE COMPANY HERE
    topics_model = sbux_train
    drilled = drill_topics(topics_model, train_preds, model)
