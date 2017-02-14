import csv
from clean_tweets import clean_pipeline
from random import shuffle

tweets = clean_pipeline('../../tweets/food')[:1000]
shuffle(tweets)

def to_csv(filename):
    with open(filename, 'wb') as f:
        wr = csv.writer(f)
        for tweet in list(set(tweets)):
            wr.writerow([tweet])

if __name__ == '__main__':
    print '# of tweets', len(tweets)
    to_csv('../../tweets/csv/food-1a.csv')
