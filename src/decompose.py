import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# load and drop NaNs
train = pd.read_csv('../../tweets/csv/train.csv', header=None)
train.dropna(inplace=True)

test = pd.read_excel('../../tweets/csv/test1.xls')
test.dropna(inplace=True)

# segment data by company
sbux_train = train[train[0].str.contains('starbucks')]
sbux_test = test[test['tweets'].str.contains('starbucks')]

n_features = 100
ngrams = [1,2]
max_df = .5
n_topics = 10
n_top_words = 10

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def tfidf(train, test):
    tf = TfidfVectorizer(stop_words='english', max_features=n_features, ngram_range=ngrams, max_df=.5)
    train = tf.fit_transform(train)
    feature_names = tf.get_feature_names()
    test = tf.transform(test)
    return train, test, feature_names


def decompose(model, train, test, feature_names):
    model.fit(train)
    print_top_words(model, feature_names, n_top_words)
    train_dist_preds = model.transform(train)
    train_preds = np.argmax(train_dist_preds, axis=1)
    test_dist_preds = model.transform(test)
    test_preds = np.argmax(test_dist_preds, axis=1)
    return train_preds, test_preds

def topic_summaries(df, test_preds, mod_name):
    df['topics'] = test_preds.tolist()
    file_dir = '../../tweets/topic_dfs/{}_topic_preds.csv'.format(mod_name)
    df.to_csv(file_dir)
    topic_df = pd.read_csv(file_dir)
    summary = topic_df.groupby('topics')['labels'].mean()
    count = topic_df.groupby(['topics','labels']).count()
    return summary, count

if __name__ == '__main__':
    train = sbux_train.values.reshape(sbux_train.values.shape[0],)
    test = sbux_test['tweets'].values.reshape(sbux_test.values.shape[0],)
    train, test, feature_names = tfidf(train, test)
    lda = LatentDirichletAllocation(n_topics=n_topics, learning_method='online', learning_offset=50.)
    nmf = NMF(n_components=n_topics, init='random')
    ###############################################
    model = nmf # Set model here
    ###############################################
    train_preds, test_preds = decompose(model, train, test, feature_names)
    summary, count = topic_summaries(sbux_test, test_preds, mod_name='nmf')
