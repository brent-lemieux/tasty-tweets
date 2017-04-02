'''
This script is under construction.  It will eventually replace plot_ae.py.
'''
from clean_tweets import clean_pipeline
from ae_km import Cluster

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from wordcloud import WordCloud, STOPWORDS
from collections import OrderedDict

def find_k(encoded_tweets, k_cluster_range):
    '''find the optimal number of topics for the autoencoder model'''
    scores = []
    for k in k_cluster_range:
        km = KMeans(n_clusters=k).fit(encoded_tweets)
        dists = km.transform(encoded_tweets)
        labels = km.labels_
        scores.append(silhouette_score(dists, labels))
    plt.plot(k_cluster_range, scores)
    plt.show()
    return zip(k_cluster_range, scores)

def create_cloud(df, company, topic_index):
    '''create word clouds for each topic'''
    tweets = df
    stop = STOPWORDS.add(company)
    by_topic = tweets[tweets['topic']==topic_index]['tweets'].tolist()
    print '# of tweets in topic {} = {}'.format(topic_index, len(by_topic))
    string = ' '.join(by_topic)
    wordcloud = WordCloud(stopwords=stop, width=800, height=600).generate(string)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../exploratory_plots/ae_{}{}_cloud.png'.format(company, topic_index))



if __name__ == '__main__':
    folder = 'snap'
    tweets = clean_pipeline('../../tweets/{}'.format(folder))
    labels = Cluster(tweets, 10)
