# Source: https://blog.keras.io/building-autoencoders-in-keras.html
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers

import pandas as pd
import numpy as np
import pickle

import logging
from gensim.models import Word2Vec


def embed_tweets(f, company, train=True):
    df = pd.read_csv(f)
    tweets = [x.split(' ') for x in df['tweets'].tolist()]
    if train:
        model = Word2Vec(tweets, size=50, min_count=20, workers=3)
    else:
        model = pickle.load(open('../models/embed_model.pkl', 'rb'))
    print model.most_similar(positive=['cheeseburger', 'chipotle'], negative=['mcdonalds'], topn=5)
    tweets = [tweet for tweet in tweets if company in tweet]
    embedded_tweets = []
    master_tweets = []
    for tweet in tweets:
        try:
            matrix = np.zeros((30, 50))
            for i, word in enumerate(tweet):
                try:
                    matrix[i,:] = model[word]
                except:
                    matrix[i,:] = np.zeros((50))
            # print matrix
            embedded_tweets.append(np.reshape(matrix,1500))
            master_tweets.append(' '.join(tweet))
        except:
            pass
    return model, embedded_tweets, master_tweets

def autoencoder(x):
    x_train = np.array(x[5000:])
    x_test = np.array(x[:5000])
    # this is the size of our encoded representations
    encoding_dims = [64, 32]  # 32 floats -> compression of factor 109, assuming the input is 3500 floats
    # this is our input placeholder
    input_tweet = Input(shape=(1500,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(64, activation='relu')(input_tweet)
    encoded = Dense(32, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(1500, activation='sigmoid')(decoded)
    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_tweet, output=decoded)
    # this model maps an input to its encoded representation
    encoder = Model(input=input_tweet, output=encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(64,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    autoencoder.fit(x_train, x_train,
                nb_epoch=30,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))
    encoded_tweets = encoder.predict(np.array(x))
    # decoded_tweets = decoder.predict(encoded_tweets)
    return encoded_tweets


if __name__ == '__main__':
    embed_model, embeddings, tweets = embed_tweets('../../tweets/csv/clean_master.csv', 'starbucks', train=False)
    encoded = autoencoder(embeddings)
    pickle.dump(embed_model, open('embed_model.pkl', 'wb'))
    pickle.dump(encoded, open('encoded_tweets.pkl', 'wb'))
    pickle.dump(tweets, open('tweets_ae.pkl', 'wb'))
