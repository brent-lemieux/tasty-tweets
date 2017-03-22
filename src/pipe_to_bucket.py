import pip

pip.main(['install', 'tweepy'])
pip.main(['install', 'boto'])


from extract_tweets import get_tweets
from tweets_to_bucket import to_bucket
import time




if __name__ == '__main__':
    date = time.strftime("%m-%d-%Y--%H-%M")
    topics = ['chipotle']
    get_tweets(topics, '/home/ubuntu/tweets/tweets-{}'.format(date), 1)
    to_bucket()
