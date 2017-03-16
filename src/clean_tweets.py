import pickle
from string import punctuation
from string import printable
import os
import spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pandas as pd

from emojis import replace_emoji, map_dic



if not 'nlp' in locals():
    print "Loading English Module..."
    nlp = spacy.load('en')
    print "Module loading complete."


def load_tweets(dirname):
    '''load all tweets and return as a set to avoid repeats'''
    tweets = []
    for pkl_file in os.listdir(dirname):
        try:
            tweets += pickle.load( open('{}/{}'.format(dirname,pkl_file), 'rb'))
        except:
            print str(pkl_file)
    return list(set(tweets))

def lemmatize(doc, stop_words=ENGLISH_STOP_WORDS):
    '''lemmatize the tweet -- returns each word to its base dictionary form'''
    doc = doc.replace('  ',' ')
    doc = unicode(doc)
    doc = nlp(doc)
    tokens = [token.lemma_ for token in doc if token not in stop_words and token]
    return ' '.join(tokens)

twitter_speak = pickle.load( open( 'twitter_speak.pkl', 'rb' ) )

def convert_slang(doc, slang_dic=twitter_speak):
    '''translate slange to "proper" english'''
    for k, v in slang_dic.iteritems():
        if k in doc:
            doc = doc.replace(k, v)
    return doc

def clean_tweets(tweets):
    '''remove all unicode, unspecified punctuation, and convert to lower case'''
    cleaner_tweets = []
    tweets, dates = zip(*tweets)
    for tweet in tweets:
        clean_tweet = []
        for char in tweet:
            if char in printable and char not in punctuation or char in ['#', '$', '@']:
                clean_tweet.append(char.lower())
        cleaner_tweets.append(''.join(clean_tweet))
    return zip(cleaner_tweets, dates)

def cleaner_tweets(tweets):
    '''remove links and retweets, as well as other special characters'''
    tweets = clean_tweets(tweets)
    tweets, dates = zip(*tweets)
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
    cleanest_tweets = [convert_slang(tweet) for tweet in cleaner_tweets]
    lemmatized = [lemmatize(tweet) for tweet in cleanest_tweets]
    return zip(lemmatized, dates)


def clean_pipeline(dirname):
    '''pipeline to clean all tweets and translate emojis'''
    tweets = load_tweets(dirname)
    tweets = replace_emoji(tweets, map_dic)
    tweets = cleaner_tweets(tweets)
    df = pd.DataFrame(tweets, columns=['tweets', 'dates'])
    df['dates'] = df['dates'].map(lambda x: str(x)[:10])
    return df

if __name__ == '__main__':
    folder = 'snap'
    tweets = clean_pipeline('../../tweets/{}'.format(folder))
