'''
This file is under construction.  The purpose is to replace the autoencoder.py
and plot_ae.py.  It will use a class that encodes tweet vectors and clusters
them together.
'''
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
import keras

import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class Cluster():
    def __init__(self, tweet_df, k):
        self.tweets = tweet_df['tweets'].values
        self.dates = tweet_df['dates'].values
        self.k = k
        self.tfidf = None
        self.encoder = None

    def vectorize(self, ngram_range=[1,2], min_df=0.001):
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, min_df=min_df)
        vecs = tfidf.fit_transform(self.tweets)
        self.tfidf = tfidf
        return vecs

    def encode(self, vecs, architecture=[60,30,60], activations=['tanh', 'sigmoid'], loss='binary_crossentropy'):
        train = vecs[5000:,:]
        test = vecs[:5000,:]
        input_cols = test.shape[1]
        input_tweet = Input(shape=(input_cols,))
        encoded = Dense(architecture[0], activation=activations[0])(input_tweet)
        encoded = Dense(architecture[1], activation=activations[0])(encoded)
        decoded = Dense(architecture[2], activation=activations[0])(encoded)
        decoded = Dense(input_cols, activation=activations[1])(decoded)
        # take in tweet and reconstruct it
        autoencoder = Model(input=input_tweet, output=decoded)
        # create encoder
        encoder = Model(input=input_tweet, output=encoded)
        # final layer encoder input shape
        encoded_input = Input(shape=(architecture[-1],))
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
        adam = keras.optimizers.Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        autoencoder.compile(optimizer=adam, loss=loss)
        autoencoder.fit(train, train,
                    nb_epoch=50,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(test, test))
        encoded_tweets = encoder.predict(np.array(self.vecs))
        self.encoder = encoder
        return encoded_tweets

    def kmeans(self):
        vecs = self.vectorize()
        encoded = self.encode(vecs)
        km = KMeans(n_clusters=self.k)
        labels = km.fit_transform(encoded).labels_
        self.km = km
        return labels
