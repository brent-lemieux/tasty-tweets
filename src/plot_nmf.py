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
from collections import OrderedDict


data = pd.read_csv('/Users/blemieux/projects/tweets/csv/clean_master.csv')

# segment data by company

# Starbucks
sbux = data[data['tweets'].str.contains('starbucks')]
# load model
model = pickle.load(open('../models/sbux_nmf.pkl', 'rb'))
# define TfidfVectorizer parameters
n_features = 1000
ngrams = [1,3]
max_df = .5
min_df = .01
n_topics = 18
n_top_words = 10


#
# # Chipotle
# cmg = data[data['tweets'].str.contains('chipotle')]
# # load model
# model = pickle.load(open('../models/cmg_nmf.pkl', 'rb'))
# n_features = 1000
# ngrams = [1,2]
# max_df = .5
# min_df = .005
# n_topics = 17
# n_top_words = 10



# # McDonalds
# mcd = data[data['tweets'].str.contains('mcdonalds')]
# # load model
# model = pickle.load(open('../models/mcd_nmf.pkl', 'rb'))
# # define TfidfVectorizer parameters
# n_features = 1000
# ngrams = [1,3]
# max_df = .5
# min_df = .01
# n_topics = 18
# n_top_words = 10


# vectorize
tfidf = TfidfVectorizer(max_features=n_features, ngram_range=ngrams, max_df=max_df, min_df=min_df,stop_words='english')

vecs = tfidf.fit_transform(sbux['tweets']).todense()
feature_names = tfidf.get_feature_names()

sbux['topic'] = np.argmax(model.transform(vecs), axis=1)
# stock = pd.read_csv('../sbux_stock.csv')
# price_delta = stock['Price Change'].tolist()[::-1]

def plot_topic_trend(df,topic_index, topic_name, vlines=[], stock=[], event=None):
    topic_share = []
    days = []
    dates = np.unique(df['date'])
    dts = [dt[-5:] for dt in dates]
    for i, date in enumerate(dates):
        df_date = df[df['date']==date]
        topic_tweets = df_date[df_date['topic'].isin(topic_index)]
        topic_share.append(len(topic_tweets)/len(df_date))
        days.append(i+1)
    plt.plot(days, topic_share, lw=2, label="Prevelance of Topic")
    if len(stock) > 0:
        plt.plot(days, stock, color='teal', label="Stock Price Change", lw=2)
    plt.xticks(days, dts, rotation='vertical')
    if len(vlines) > 0:
        [plt.axvline(x, color='grey', label='Weekend day') for x in vlines]
    if event:
        plt.axvline(event[0], color='navy', lw=2, label=event[1], alpha=.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.subplots_adjust(bottom=0.15)
    plt.title('{} Topic Overtime'.format(topic_name))
    plt.ylabel('Percentage of tweets about topic')
    plt.xlabel('Day')
    plt.savefig('../plots/{}_topic_ts.png'.format(topic_name.replace(' ','_')))
    plt.show()


def create_cloud(df, topic_index, extra_stop, topic_name):
    for word in extra_stop:
        STOPWORDS.add(word)
    tweets = df[df['topic'].isin(topic_index)]['tweets'].tolist()
    tweets = ' '.join(tweets)
    wordcloud = WordCloud(stopwords=STOPWORDS, width=800,
     height=600).generate(tweets)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../plots/{}_cloud.png'.format(topic_name.replace(' ','_')))
    plt.show()

weekends = [10,11,17,18]

def get_labeled_topics(model, tfidf, company):
    from load_and_process import load_xls
    df1 = load_xls('../../tweets/csv/test1.xls', slang=False, lemma=True, pos=False)
    df1 = df1[df1['tweets'].str.contains(company)]
    df2 = load_xls('../../tweets/csv/test2.xls', slang=False, lemma=True, pos=False)
    df2 = df2[df2['tweets'].str.contains(company)]
    # df = pd.concat([df1, df2])
    X1 = tfidf.transform(df1['tweets'])
    X2 = tfidf.transform(df2['tweets'])
    topics = np.argmax(model.transform(X1), axis=1).tolist(), np.argmax(model.transform(X2), axis=1).tolist()
    df1['topics'] = topics[0]
    df2['topics'] = topics[1]
    df1['labels'] = df1['labels'] - 2
    df2['labels'] = df2['labels'] - 2
    return df1, df2

def topic_distributions(df1, df2, company, topic_index):
    topic_df1 = df1[df1['topics'].isin(topic_index)]
    topic_df2 = df2[df2['topics'].isin(topic_index)]
    topic1 = topic_df1.groupby('labels')['tweets'].agg(['count'])
    topic2 = topic_df2.groupby('labels')['tweets'].agg(['count'])
    topic1['pct_of_tweets'] = topic1['count'] / len(topic_df1)
    topic2['pct_of_tweets'] = topic2['count'] / len(topic_df2)
    print topic1, topic2
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    seaborn.barplot(df1['lab'], df1['pct_of_tweets'])
    seaborn.barplot(df2['lab'], df2['pct_of_tweets'])
    plt.show()

if __name__ == '__main__':
    plt.close('all')
    # plt.style.use('ggplot')
    topic = 'Chipotle Guacamole'
    topic_idx = [12, 14]
    co = sbux
    event1 = [2, 'Refugee Hiring Announcement']
    event2 = [13, "Valentine's Day"]
    # plot_topic_trend(co, topic_idx, topic, vlines=weekends)
    # plt.close('all')
    # create_cloud(co, topic_idx, ['chipotle'], topic)

    df1, df2 = get_labeled_topics(model, tfidf, 'starbucks')
    topic_distributions(df1, df2, 'starbucks', topic_idx)
