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
    # This function creates word embeddings for each tweets.  It returns the
    # embdding model as well as the embedded tweets.  It also returns a
    # dataframe with the raw tweet and date for each tweet that has embeddings.
    df = pd.read_csv(f)
    tweets = [x.split(' ') for x in df['tweets'].tolist()]
    dates = df['date'].tolist()
    if train:
        model = Word2Vec(tweets, size=100, min_count=20, workers=3)
    else:
        model = pickle.load(open('../models/embed_model.pkl', 'rb'))
    tweets = [tweet for tweet in tweets if company in tweet]
    embedded_tweets = []
    master_tweets = []
    master_dates = []
    for i, tweet in enumerate(tweets):
        try:
            matrix = np.zeros((30, 100))
            for i, word in enumerate(tweet):
                try:
                    matrix[i,:] = model[word]
                except:
                    matrix[i,:] = np.zeros((100))
            # print matrix
            embedded_tweets.append(np.reshape(matrix,3000))
            master_tweets.append(' '.join(tweet))
            master_dates.append(dates[i])
        except:
            pass
    # New dataframe must be created to be properly indexed with embeddings
    # matrix
    df = pd.DataFrame({'tweets':master_tweets, 'date':master_dates})
    return model, embedded_tweets, df

def autoencoder(x):
    x_train = np.array(x[5000:])
    x_test = np.array(x[:5000])
    # input shape
    input_tweet = Input(shape=(3000,))
    # 32 floats -> compression of factor 60, assuming the input is 3000 floats
    encoded = Dense(50, activation='relu')(input_tweet)
    encoded = Dense(25, activation='relu')(encoded)
    # reconstruct the input
    decoded = Dense(50, activation='relu')(encoded)
    decoded = Dense(3000, activation='sigmoid')(decoded)
    # model input to its reconstruction
    autoencoder = Model(input=input_tweet, output=decoded)
    # model input to its encoded representation
    encoder = Model(input=input_tweet, output=encoded)
    # final layer encoder input shape
    encoded_input = Input(shape=(50,))
    # setup decoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                nb_epoch=10,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test))
    encoded_tweets = encoder.predict(np.array(x))
    return encoded_tweets


if __name__ == '__main__':
    embed_model, embeddings, df = embed_tweets('../../tweets/csv/clean_master.csv', 'starbucks', train=False)
    encoded = autoencoder(embeddings)
    pickle.dump(embed_model, open('embed_model.pkl', 'wb'))
    pickle.dump(encoded, open('encoded_tweets.pkl', 'wb'))
    pickle.dump(df, open('tweets_ae.pkl', 'wb'))
    print embed_model.most_similar(positive=['cheeseburger', 'chipotle'], negative=['mcdonalds'], topn=5)
