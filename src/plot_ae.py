from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from wordcloud import WordCloud, STOPWORDS
from collections import OrderedDict


def find_k(encoded_tweets, k_cluster_range):
    scores = []
    for k in k_cluster_range:
        km = KMeans(n_clusters=k).fit(encoded_tweets)
        dists = km.transform(encoded_tweets)
        labels = km.labels_
        scores.append(silhouette_score(dists, labels))
    plt.plot(k_cluster_range, scores)
    plt.show()
    return zip(k_cluster_range, scores)


def kmeans(encoded_tweets, k, company):
    encoded_tweets = np.array(encoded_tweets)
    print encoded_tweets.shape
    km = KMeans(n_clusters=k).fit(encoded_tweets)
    labels = km.labels_
    print np.unique(labels)
    tweets = pickle.load(open('cmg_tweets_ae.pkl', 'rb'))
    dates = pickle.load(open('cmg_dates_ae.pkl', 'rb'))
    df = pd.DataFrame({'tweets':tweets,
                        'date':dates,
                        'topic':labels})
    return df, km


# # Get stock data
stock_file = '../stock_data/cmg_stock.csv'
stock_price = pd.read_csv(stock_file)
stock_price['Description'] = 'Stock Price Change'
# price_delta = stock['Stock Price Change'].tolist()[::-1]


def plot_topic_trend(df, topic_index, topic_name, vday=None, stock=[], refugee=None):
    topic_share = []
    days = []
    dates = np.unique(df['date'])
    dts = [dt[-5:] for dt in dates]
    for i, date in enumerate(dates):
        df_date = df[df['date']==date]
        topic_tweets = df_date[df_date['topic'].isin(topic_index)]
        topic_share.append(len(topic_tweets)/len(df_date))
        days.append(i+1)
    plt.plot(days, topic_share, lw=2, label="Prevalance of Topic", color='navy')
    # if len(stock) > 0:
    #     plt.plot(days, stock, color='teal', label="Stock Price Change", lw=2)
    plt.xticks(days, dts, rotation='vertical')
    plt.ylim((-.05,.35))
    if vday:
        plt.axvline(vday[0], color='red', lw=2, label=vday[1], alpha=.5)
    if refugee:
        plt.axvline(refugee[0], color='green', lw=2, label=refugee[1], alpha=.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.subplots_adjust(bottom=0.15)
    plt.title('{} Topic Overtime'.format(topic_name))
    plt.ylabel('Percentage of tweets about topic')
    plt.xlabel('Day')
    plt.savefig('../exploratory_plots/{}_topic_ts.png'.format(topic_name.replace(' ','_')))
    plt.show()
    # Uncomment below to create csv for d3 plot
    df1 = pd.DataFrame({'Date':dates, 'Data':topic_share})
    df1['Description'] = 'Topic Prevalance'
    df = pd.concat([df1, stock])
    df.to_csv('../final_plots/cmg{}.csv'.format(topic_index[0]))


def create_cloud(df, company, k, topic_index):
    tweets = df
    stop = STOPWORDS.add(company)
    by_topic = tweets[tweets['topic']==topic_index]['tweets'].tolist()
    print '# of tweets in topic {} = {}'.format(topic_index, len(by_topic))
    string = ' '.join(by_topic)
    wordcloud = WordCloud(stopwords=stop, width=800, height=600).generate(string)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('../exploratory_plots/ae_{}{}_cloud.png'.format(company, topic_index))


def get_labeled_topics(encoder, km, company):
    # create a DataFrame with topics and sentiment labels by applying model
    # to labeled subset
    from load_and_process import load_xls
    df1 = load_xls('../../tweets/csv/test1.xls', slang=True, lemma=True, pos=False)
    df2 = load_xls('../../tweets/csv/test2.xls', slang=True, lemma=True, pos=False)
    df = pd.concat([df1, df2])
    df = df[df['tweets'].str.contains(company)]
    # df = pd.concat([df1, df2])
    tfidf = pickle.load(open('tfidf_ae.pkl', 'rb'))
    print tfidf
    X = tfidf.transform(df['tweets'].values).todense()
    topics = km.predict(encoder.predict(X))
    df['topics'] = topics
    df['labels'] = df['labels'] - 2
    return df


def topic_distributions(df, topic_name, topic_index):
    topic_df = df[df['topics'].isin(topic_index)]
    topic = topic_df.groupby('labels')['tweets'].agg(['count'])
    topic['pct_of_tweets'] = topic['count'] / len(topic_df)
    colors = ['pale red', 'greyish', 'windows blue']
    current_palette_3 = sns.xkcd_palette(colors)
    sns.set_palette(current_palette_3)
    sns.set_style('darkgrid')
    sns.barplot(topic.index, topic['pct_of_tweets'])
    plt.xticks([0,1,2], ['Negative', 'Neutral', 'Positive'])
    plt.ylabel('Percent of Tweets in Topic')
    plt.xlabel('Brand Sentiment')
    plt.ylim((0.0,1.0))
    plt.title('{}_sentiment'.format(topic_name))
    plt.savefig('../exploratory_plots/{}_sentiment.png'.format(topic_name))
    plt.show()
    return topic_df, topic


def pca(vecs, labels):
    pcs = PCA(n_components=2).fit_transform(vecs)
    df = pd.DataFrame({'x':list(pcs[:,0].flatten()), 'y':list(pcs[:,1].flatten()), 'topics':labels})
    df = df[df['topics'] < 6]
    sns.lmplot('x', 'y', data=df, hue='topics', fit_reg=False, size=4, aspect=2)
    plt.subplots_adjust(top=.8)
    plt.title('Autoencoder PCA Topic Separation')
    plt.savefig('../exploratory_plots/pca_ae_plot.png')
    plt.show()


def create_plots(vecs, k, company, encoder):
    df, km = kmeans(vecs, k, company)
    # labels = km.labels_
    # pca(vecs, labels)
    event1 = [2, 'Refugee Hiring Announcement']
    event2 = [12, "Valentine's Day"]
    # topic = 'Starbucks Refugee Topic Prevalance'
    # plot_topic_trend(df, [1], topic, vday=event2, refugee=event1)
    sent_df = get_labeled_topics(encoder, km, company)
    dists = []
    for x in range(k):
        plt.close('all')
        topic = 'ae_Chipotle{}'.format(x)
        create_cloud(df, company, k, x)
        plt.close('all')
        plot_topic_trend(df, [x], topic, vday=event2, stock=stock_price)
        plt.close('all')
        try:
            topic_df, topic_dist = topic_distributions(sent_df, topic, [x])
            dists.append((topic_df, topic_dist))
        except:
            pass
    return dists

if __name__ == '__main__':
    plt.close('all')
    plt.style.use('ggplot')
    vecs = pickle.load(open('cmg_encoded_tweets.pkl', 'rb'))
    company = 'chipotle'
    k = 15
    encoder = pickle.load(open('../models/cmg_encoder.pkl', 'rb'))
    # embedder = pickle.load(open('../models/embed_model.pkl', 'rb'))
    dists = create_plots(vecs, k, company, encoder)
