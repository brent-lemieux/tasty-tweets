import pickle
from string import punctuation
from string import printable
import os
from emotis import replace_emoji, map_dic


def load_tweets(dirname):
    tweets = []
    for pkl_file in os.listdir(dirname):
        try:
            tweets += list(pickle.load( open('/{}/{}'.format(dirname,pkl_file), 'rb')))
        except:
            print str(pkl_file)
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
            if 'http' not in word and word != 'rt':
                clean_tweet.append(word)
        cleaner_tweets.append(' '.join(clean_tweet))
    cleaner_tweets = [' '.join(tweet.split('\n')) for tweet in cleaner_tweets]
    cleaner_tweets = [' '.join(tweet.split(' amp ')) for tweet in cleaner_tweets]
    return cleaner_tweets


def clean_pipeline(dirname):
    master = []
    tweets = load_tweets(dirname)
    tweets = replace_emoji(tweets, map_dic)
    tweets = cleaner_tweets(tweets)
    return tweets
