import gensim

def embed_tweets(tweets):
    tokens = [tweet.split(' ') for tweet in tweets]
    model = gensim.models.Word2Vec(tokens)
    return model
