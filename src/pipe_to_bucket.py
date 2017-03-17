from extract_tweets import get_tweets
from tweets_to_bucket import to_bucket







if __name__ == '__main__':
    date = time.strftime("%m-%d-%Y--%H-%M")
    topics = ['chipotle']
    get_tweets(topics, '../../tweets/tweets-{}'.format(date), 1)
    to_bucket()
