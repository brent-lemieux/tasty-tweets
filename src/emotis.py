import re
import pickle
from collections import defaultmap_dict
import os
# from sys import argv
import pandas as pd

# label = argv
# prompt = '> '

# tweets = list(pickle.load( open('food/food0208a.pkl', 'rb')))

df = pd.DataFrame(pickle.load( open('df_emojis.pkl', 'rb')))

map_dic = {v:k for k, v in zip(list(df['short_name']), list(df['unichar']))}

def load_tweets(dirname):
    tweets = []
    for pkl_file in os.listdir(dirname):
        tweets += list(pickle.load( open('{}/{}'.format(dirname,pkl_file), 'rb')))
    return list(set(tweets))

def replace_emoji(tweets, map_dic):
    cts = []
    for tweet in tweets:
        emoji = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        for e in emoji.findall(tweet):
            if e in map_dic.keys():
                tweet = tweet.replace(e,' {} '.format(map_dic[e]))
            else:
                tweet = tweet.replace(e,' ')
        cts.append(tweet)
    # Uncomment below to turn emojis into english
    # cts = [ct.replace('_',' ') for ct in cts]
    return cts


def find_emoji(tweets):
    emojis = []
    for tweet in tweets:
        emoji = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        emojis += emoji.findall(tweet)
    return emojis

def emoji_counter(emojis):
    done = []
    counts = []
    for e in emojis:
        if e not in done:
            counts.append([e, emojis.count(e)])
            done.append(e)
    return sorted(counts, key = lambda x: x[1])[::-1]


if __name__ == '__main__':
    tweets = load_tweets('food')
    emojis, emojiproofed = find_emoji(tweets, map_dic)
    # counts = emoji_counter(emojis)
