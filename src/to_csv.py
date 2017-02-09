import csv
from clean_tweets import clean_pipeline, food_files



tweets = clean_pipeline('food')


def to_csv(filename):
    with open(filename, 'wb') as f:
        wr = csv.writer(f)
        for tweet in list(set(tweets)):
            wr.writerow([tweet])

if __name__ == '__main__':
    print '# of tweets', len(tweets)
    to_csv('food.csv')
