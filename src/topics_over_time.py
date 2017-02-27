# Track prevelance of negative topics overtime

# Negative topics model 1 = (refugees: 8), (refugees: 10),
# (expensive, cup: 12), (refugees: 15)

# Negative topics model 2 = (refugees: 8), (refugees: 17),
# (expensive, cup: 12)

# Negative topics model 3 = (refugees: 5), (refugees: 6),
# (expensive: 13)
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from decompose import print_top_words
from wordcloud import WordCloud, STOPWORDS


data = pd.read_csv('/Users/blemieux/projects/tweets/csv/clean_master.csv')

# segment data by company

# Starbucks
sbux = data[data['tweets'].str.contains('starbucks')]
# load model
model = pickle.load(open('../models/sbux_nmf3.pkl', 'rb'))
# define TfidfVectorizer parameters
n_features = 400
ngrams = [1,3]
max_df = .5
min_df = .01
# define # of top words to show
n_top_words = 10

# vectorize
tfidf = TfidfVectorizer(max_features=n_features, ngram_range=ngrams, max_df=max_df, min_df=min_df,stop_words='english')

vecs = tfidf.fit_transform(sbux['tweets']).todense()
feature_names = tfidf.get_feature_names()

sbux['topic'] = np.argmax(model.transform(vecs), axis=1)

refugee_topics = [5, 6]
cup_topic = [13]

def plot_topic_trend(df, topic_index, topic_name):
    topic_share = []
    days = []
    dates = np.unique(df['date'])
    for i, date in enumerate(dates):
        df_date = sbux[sbux['date']==date]
        num_tweets = len(df_date)
        topic_tweets = df_date[df_date['topic'].isin(topic_index)]
        num_topic_tweets = len(topic_tweets)
        topic_share.append(num_topic_tweets/num_tweets)
        days.append(i+1)
    plt.plot(days, topic_share, lw=3)
    plt.title('{} Topic Overtime'.format(topic_name))
    plt.ylabel('Percentage of tweets about topic')
    plt.xlabel('Day')
    plt.savefig('{}_topic_ts.png'.format(topic_name.replace(' ','_')))
    plt.show()

extra_stop = ['starbucks']

def create_cloud(df, topic_index, extra_stop):
    for word in extra_stop:
        STOPWORDS.add(word)
    tweets = df[df['topic'].isin(topic_index)]['tweets'].tolist()
    tweets = ' '.join(tweets)
    wordcloud = WordCloud(stopwords=STOPWORDS, width=800,
     height=600).generate(tweets)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    plt.close('all')
    plt.style.use('ggplot')
    # plot_topic_trend(sbux, refugee_topics, 'Refugee')
    # plt.close('all')
    # plot_topic_trend(sbux, [8], 'Valentines Day')
    create_cloud(sbux, refugee_topics, extra_stop)
