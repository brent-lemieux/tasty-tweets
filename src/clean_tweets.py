import pickle
import spacy
from string import punctuation
from string import printable
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import os
from emotis import replace_emoji, map_dic
### consider using fuzzywuzzy in cleaning pipeline

def load_tweets(dirname):
    tweets = []
    for pkl_file in os.listdir(dirname):
        tweets += list(pickle.load( open('{}/{}'.format(dirname,pkl_file), 'rb')))
    return list(set(tweets))

def clean_tweets(tweets):
    cleaner_tweets = []
    for tweet in tweets:
        clean_tweet = []
        for char in tweet:
            if char in printable and char not in punctuation or char == '#' or char == '$' or char == '@':
                clean_tweet.append(char.lower())
        cleaner_tweets.append(''.join(clean_tweet))
    return cleaner_tweets

def cleaner_tweets(tweets):
    tweets = clean_tweets(tweets)
    cleaner_tweets = []
    lst_tweets = [tweet.split(' ') for tweet in tweets]
    for tweet in lst_tweets:
        clean_tweet = []
        for word in tweet:
            if word[:4] != 'http' and word != 'rt':
                clean_tweet.append(word)
        cleaner_tweets.append(' '.join(clean_tweet))
    cleaner_tweets = [' '.join(tweet.split('\n')) for tweet in cleaner_tweets]
    cleaner_tweets = [' '.join(tweet.split('amp')) for tweet in cleaner_tweets]
    return cleaner_tweets

# if not 'nlp' in locals():
#     print "Loading English Module..."
#     nlp = spacy.load('en')
#
# STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n't", "'s'", "'m'"])
#
# def lemmatize_string(doc, stop_words):
#     doc = unicode(doc.translate(None, punctuation))
#     doc = nlp(doc)
#     tokens = [token.lemma_ for token in doc]
#     return ' '.join(w for w in tokens if w not in stop_words)

def clean_pipeline(dirname):
    master = []
    tweets = load_tweets(dirname)
    tweets = replace_emoji(tweets, map_dic)
    tweets = cleaner_tweets(tweets)
    # lemma = [lemmatize_string(str(tweet), STOPLIST) for tweet in tweets]
    return tweets

trump_files = ['data/trump_tweets_1-21-17.pkl', 'data/trump_tweets_1-22-17.pkl', 'data/trump_tweets_1-22-17b.pkl', 'data/trump_tweets_1-23-17a.pkl', 'data/trump_tweets_1-23-17b.pkl', 'data/trump_tweets_1-23-17c.pkl', 'data/trump_tweets_1-23-17d.pkl', 'data/trump_tweets_1-24-17a.pkl', 'data/trump_tweets_1-24-17b.pkl']

econ = ['financial_tweets/econ_tweets_0124a.pkl', 'financial_tweets/econ_tweets_0124b.pkl', 'financial_tweets/econ_tweets_0124c.pkl']

food_files = ['food/chip_v_qdob.pkl','food/chip_v_qdob2.pkl','food/chip_v_qdob3.pkl', 'food/chip_v_qdob4.pkl']



if __name__ == '__main__':
    tweets = clean_pipeline(['data/trump_tweets_0201.pkl'])
