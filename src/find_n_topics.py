# This script plots reconstruction error for
# the decomposition model in order to determine
# the optimal number of topics

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_csv('/Users/blemieux/projects/tweets/csv/clean_master.csv')

company = 'starbucks'

df = data[data['tweets'].str.contains(company)]


# Set tfidf/bow and decomposition parameters
n_features = 1000
ngrams = [1,3]
max_df = .5
min_df = .01
n_top_words = 10


tf = TfidfVectorizer(max_features=n_features, ngram_range=ngrams, max_df=max_df, min_df=min_df, stop_words='english')
X = tf.fit_transform(df['tweets']).todense()
feature_names = tf.get_feature_names()


def print_top_words(model, feature_names, n_top_words):
    # show top words for each cluster
    words = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def find_k_clusters(ks, tf, X):
    recon_errors = []
    for k_clusters in ks:
        print k_clusters, 'Clusters'
        # Initialize the model with n_clusters value and a random generator
        # seed of 4 for reproducibility.
        # Create a number of models to test out
        nmf = NMF(n_components=k_clusters, init='random', random_state=4)
        #############################################
        model = nmf # Set model here
        mod_name = 'nmf'
        #############################################
        model.fit(X)
        labs = model.transform(X)
        recon_errors.append(model.reconstruction_err_)
        print 'Recontruction Error for', k_clusters, 'Latent Topics =', model.reconstruction_err_
    sns.set_style("darkgrid")
    plt.plot(ks, recon_errors, lw=3 )
    plt.title('{} RECONSTRUCTION ERROR FOR EACK K CLUSTERS'.format(company.upper()))
    plt.xlabel('k Clusters')
    plt.ylabel('Recontruction Error')
    plt.savefig('../plots/{}_n_cluster_finder.png'.format(company))
    plt.close('all')
    return ks[np.argmin(recon_errors)]


if __name__ == '__main__':
    plt.close('all')
    ks = [10,15,20,25,30]
    k = find_k_clusters(ks, tf, X)
