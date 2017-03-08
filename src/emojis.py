import re
import pickle
import os

import pandas as pd

# Uncomment line below to load emoji DataFrame
# tweets = list(pickle.load( open('food/food0208a.pkl', 'rb')))

df = pd.DataFrame(pickle.load( open('df_emojis.pkl', 'rb')))

# Create a dictionary to map emojis to a description
map_dic = {v:k for k, v in zip(list(df['short_name']), list(df['unichar']))}

def load_tweets(dirname):
    tweets = []
    for pkl_file in os.listdir(dirname):
        tweets += list(pickle.load( open('{}/{}'.format(dirname,pkl_file), 'rb')))
    return list(set(tweets))

def replace_emoji(tweets, map_dic):
    '''replace emojis with a description of said emoji'''
    translated_tweets = []
    tweets, dates = zip(*tweets)
    for tweet in tweets:
        emoji = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        for e in emoji.findall(tweet):
            if e in map_dic.keys():
                tweet = tweet.replace(e,' {} '.format(map_dic[e]))
            else:
                tweet = tweet.replace(e,' ')
        translated_tweets.append(tweet)
    # Uncomment below to turn underscored emoji descriptions into words
    # translated_tweets = [tweet.replace('_',' ') for tweet in translated_tweets]
    return zip(translated_tweets, dates)
