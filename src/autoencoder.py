import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from preprocess import vectorize_tweets


max_words = 35
top_words = 10000
print 'Loading tweets...'
X = vectorize_tweets('../../tweets/csv/train.csv', max_words=max_words)
x_train = X[:int(len(X)*.75),:]
x_test = X[int(len(X)*.75):,:]

vec_length = 32
encoding_dim = 30 # reduce from 35 X 32


#input placeholder
# input_tweet = Input(shape=(vec_length*max_words,))
embedding = Embedding(top_words, vec_length, max_words)
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(embedding)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(1120, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input=input_tweet, output=decoded)
# add embedding layer
print autoencoder

# autoencoder.add(Embedding(top_words, vec_length, input_length=max_words))
# autoencoder.add(Flatten())

# this model maps an input to its encoded representation
encoder = Model(input=input_tweet, output=encoded)
print encoder

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
print 'Train model...'
autoencoder.fit(x_train, x_train,
            nb_epoch=50,
            batch_size=200,
            shuffle=True,
            validation_data=(x_test, x_test))
