from extract_tweets import get_tweets
from tweets_to_bucket import to_bucket







if __name__ == '__main__':
    date = time.strftime("%m-%d-%Y--%H-%M")
    topics = ['chipotle', 'starbucks', 'mcdonalds']
    get_tweets(topics, '../../tweets/tweets-{}'.format(date), 500)
    to_bucket()
