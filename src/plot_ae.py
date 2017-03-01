from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from wordcloud import WordCloud, STOPWORDS


def find_k(encoded_tweets, k_cluster_range):
    scores = []
    for k in k_cluster_range:
        km = KMeans(n_clusters=k).fit(encoded_tweets)
        dists = km.transform(encoded_tweets)
        labels = km.labels_
        scores.append(silhouette_score(dists, labels))
    plt.plot(k_cluster_range, scores)
    plt.show()
    return zip(k_cluster_range, scores)

def kmeans(encoded_tweets, k, company):
    km = KMeans(n_clusters=k).fit(encoded_tweets)
    labels = km.labels_
    tweets = pickle.load(open('tweets_ae.pkl', 'rb'))
    df = pd.DataFrame({'tweets':tweets, 'topic':labels})
    return df

if __name__ == '__main__':
    plt.close('all')
    plt.style.use('ggplot')
    vecs = pickle.load(open('encoded_tweets.pkl', 'rb'))
    # ks = range(2,21)
    # scores = find_k(vecs, ks)
    tweets = kmeans(vecs, 12, 'starbucks')
    stop = STOPWORDS.add('starbucks')
    for x in np.unique(tweets['topic']):
        by_topic = tweets[tweets['topic']==x]['tweets'].tolist()
        print '# of tweets in topic {} = {}'.format(x, len(by_topic))
        string = ' '.join(by_topic)
        wordcloud = WordCloud(stopwords=stop, width=800,
         height=600).generate(string)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.savefig('../plots/ae_k_{}_cloud.png'.format(x))
