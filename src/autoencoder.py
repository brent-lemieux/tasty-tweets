# Source: https://blog.keras.io/building-autoencoders-in-keras.html
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
import keras

import pandas as pd
import numpy as np
import pickle

import logging
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def embed_tweets(f, company, train=True):
    # This function creates word embeddings for each tweets.  It returns the
    # embdding model as well as the embedded tweets.  It also returns a
    # dataframe with the raw tweet and date for each tweet that has embeddings.
    df = pd.read_csv(f)
    print df.values.shape
    tweets = [x.split(' ') for x in df['tweets'].tolist()]
    dates = df['date'].tolist()
    if train:
        model = Word2Vec(tweets, size=60, min_count=20, workers=3)
    else:
        model = pickle.load(open('../models/embed_model.pkl', 'rb'))
    tweets_dt = [(tweet, date) for tweet, date in zip(tweets, dates) if company in tweet]
    print len(tweets_dt)
    embedded_tweets = []
    master_tweets = []
    master_dates = []
    for tweet, date in tweets_dt:
        try:
            matrix = np.zeros((30, 60))
            for idx, word in enumerate(tweet):
                try:
                    matrix[idx,:] = model[word]
                except:
                    matrix[idx,:] = np.zeros((60))
            embedded_tweets.append(np.reshape(matrix,1800))
            # print np.max(np.reshape(matrix,3000))
            master_tweets.append(' '.join(tweet))
            master_dates.append(date)
        except:
            pass
    return model, embedded_tweets, master_tweets, master_dates

def tfidf(f, company):
    df = pd.read_csv(f)
    tweets = df[df['tweets'].str.contains(company)]['tweets'].values
    tf = TfidfVectorizer(stop_words='english', min_df=.001)
    tf.fit(tweets)
    pickle.dump(tf, open('tfidf_ae.pkl', 'wb'))
    vecs = tf.transform(tweets).todense()
    tweets = list(tweets)
    dates = df[df['tweets'].str.contains(company)]['date'].tolist()
    return vecs, tweets, dates


def autoencoder(x):
    x_train = x[5000:,:]
    x_test = x[:5000,:]
    input_cols= x.shape[1]
    # input shape
    input_tweet = Input(shape=(input_cols,))
    encoded = Dense(60, activation='tanh')(input_tweet)
    encoded = Dense(30, activation='tanh')(encoded)
    # reconstruct the input
    decoded = Dense(60, activation='tanh')(encoded)
    decoded = Dense(input_cols, activation='sigmoid')(decoded)
    # model input to its reconstruction
    autoencoder = Model(input=input_tweet, output=decoded)
    # model input to its encoded representation
    encoder = Model(input=input_tweet, output=encoded)
    # final layer encoder input shape
    encoded_input = Input(shape=(60,))
    # setup decoder model
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
    # embed_model, embeddings, tweets, dates = embed_tweets('../../tweets/csv/clean_master.csv', 'starbucks', train=False)
    # pickle.dump(embed_model, open('../models/embed_model.pkl', 'wb'))
    tf_vecs, tweets, dates = tfidf('../../tweets/csv/clean_master.csv', 'starbucks')
    pickle.dump(tweets, open('sb_tweets_ae.pkl', 'wb'))
    pickle.dump(dates, open('sb_dates_ae.pkl', 'wb'))
    encoded, encoder = autoencoder(tf_vecs)
    pickle.dump(encoded, open('sb_encoded_tweets.pkl', 'wb'))
    pickle.dump(encoder, open('../models/sb_encoder.pkl', 'wb'))
    # print embed_model.most_similar(positive=['shake', 'starbucks'], negative=['mcdonalds'], topn=5)
