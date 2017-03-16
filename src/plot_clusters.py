from clean_tweets import clean_pipeline
from ae_km import Cluster

import matplotlib.pyplot as plt

if __name__ == '__main__':
    folder = 'snap'
    tweets = clean_pipeline('../../tweets/{}'.format(folder))
    labels = Cluster(tweets, 10)
