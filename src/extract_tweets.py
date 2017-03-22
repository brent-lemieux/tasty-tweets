from __future__ import unicode_literals
import os
import tweepy
import pickle
import time

try:
    consumer_key = os.environ['TWITTER_KEY']
    consumer_secret = os.environ['TWITTER_SECRET']
    access_token = os.environ['TWITTER_AT']
    access_secret = os.environ['TWITTER_TS']
except:
    import json
    with open('/home/ubuntu/twitter.json') as key_file:
        keys = json.load(key_file)
        consumer_key = keys['TWITTER_KEY']
        consumer_secret = keys['TWITTER_SECRET']
        access_token = keys['TWITTER_AT']
        access_secret = keys['TWITTER_TS']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)


api = tweepy.API(auth)


def get_tweets(topics, save_file_name, num_batches=25):
    '''extract and pickle tweets for specified topics using Twitter's API'''
    tweets = set()
    # public_tweets = api.home_timeline()
    for i in xrange(num_batches):
        for topic in topics:
            try:
                print ('Loading', i+1, 'of', num_batches, 'batches of', topic, 'tweets')
                for tweet in tweepy.Cursor(api.search, q=topic).items(100):
                    if tweet.lang == 'en':
                        tweets.add((tweet.text, tweet.created_at))
                time.sleep(35)
            except:
                print ('Waiting for API to allow more calls...')
                time.sleep(10)
                pass
    tweets = list(tweets)
    pickle.dump( tweets, open( "{}.pkl".format(save_file_name), "wb" ) )
    print ('Succesfully pickled', len(tweets), 'tweets!')
    return tweets
