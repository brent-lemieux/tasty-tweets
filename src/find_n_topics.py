# the source for most of this code is:
# http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
from __future__ import division

from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

data = pd.read_csv('/Users/blemieux/projects/tweets/csv/clean_master.csv')

sbux = data[data['tweets'].str.contains('starbucks')]


# Set tfidf/bow and decomposition parameters
n_features = 200
ngrams = [1,3]
max_df = .5
min_df = .03
n_top_words = 10


tf = TfidfVectorizer(max_features=n_features, ngram_range=ngrams, max_df=max_df, min_df=min_df, stop_words='english')
X = tf.fit_transform(sbux['tweets']).todense()
feature_names = tf.get_feature_names()


def print_top_words(model, feature_names, n_top_words):
    # show top words for each cluster
    words = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

range_n_clusters = range(5,25)

for n_clusters in range_n_clusters:
    print n_clusters, 'Clusters'
    # Initialize the model with n_clusters value and a random generator
    # seed of 4 for reproducibility.
    # Create a number of models to test out
    lda = LatentDirichletAllocation(n_topics=n_clusters, learning_method='online', learning_offset=50., random_state=4)
    nmf = NMF(n_components=n_clusters, init='random', random_state=4)
    lsa = TruncatedSVD(n_components=n_clusters, algorithm='randomized', random_state=4)

    ###############################################
    model = nmf # Set model here
    mod_name = 'nmf'
    ###############################################

    model.fit(X)

    labs = model.transform(X)

    print 'Recontruction Error for', n_clusters, 'Latent Topics =', model.reconstruction_err_
