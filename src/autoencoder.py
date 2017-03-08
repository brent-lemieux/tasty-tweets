# Source: https://blog.keras.io/building-autoencoders-in-keras.html
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
import keras

import pandas as pd
import numpy as np
import pickle

import logging
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(f, company):
    '''get tfidf vectors and return the associated tweets and dates'''
    df = pd.read_csv(f)
    tweets = df[df['tweets'].str.contains(company)]['tweets'].values
    tf = TfidfVectorizer(stop_words='english', min_df=.001, ngram_range=[1,2])
    tf.fit(tweets)
    pickle.dump(tf, open('tfidf_ae.pkl', 'wb'))
    vecs = tf.transform(tweets).todense()
    tweets = list(tweets)
    dates = df[df['tweets'].str.contains(company)]['date'].tolist()
    return vecs, tweets, dates


def autoencoder(x):
    '''encode tfidf vectors into lower dimensional space'''
    x_train = x[5000:,:]
    x_test = x[:5000,:]
    input_cols= x.shape[1]
    input_tweet = Input(shape=(input_cols,))
    encoded = Dense(60, activation='tanh')(input_tweet)
    encoded = Dense(30, activation='tanh')(encoded)
    decoded = Dense(60, activation='tanh')(encoded)
    decoded = Dense(input_cols, activation='sigmoid')(decoded)
    # take in tweet and reconstruct it
    autoencoder = Model(input=input_tweet, output=decoded)
    # create encoder
    encoder = Model(input=input_tweet, output=encoded)
    # final layer encoder input shape
    encoded_input = Input(shape=(60,))
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    adam = keras.optimizers.Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))
    encoded_tweets = encoder.predict(np.array(x))
    return encoded_tweets, encoder


if __name__ == '__main__':
    tf_vecs, tweets, dates = tfidf('../../tweets/csv/clean_master.csv', 'starbucks')
    pickle.dump(tweets, open('../../tweets/for_model/sb_tweets_ae.pkl', 'wb'))
    pickle.dump(dates, open('sb_dates_ae.pkl', 'wb'))
    encoded, encoder = autoencoder(tf_vecs)
    pickle.dump(encoded, open('../../tweets/for_model/sb_encoded_tweets.pkl', 'wb'))
    pickle.dump(encoder, open('../models/sb_encoder.pkl', 'wb'))
