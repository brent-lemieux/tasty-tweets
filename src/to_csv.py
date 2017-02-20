from __future__ import unicode_literals
import csv
from clean_tweets import clean_pipeline
from random import shuffle

tweets = clean_pipeline('/Users/blemieux/projects/tweets/food')
shuffle(tweets)

def to_csv(filename, tweets):
    with open(filename, 'wb') as f:
        wr = csv.writer(f)
        for tweet in list(set(tweets)):
            wr.writerow([tweet])

if __name__ == '__main__':
    print '# of tweets', len(tweets)
    test = tweets[:10000]
    to_csv('../../tweets/csv/test.csv', test)
    train = tweets[10000:]
    to_csv('../../tweets/csv/train.csv', train)
