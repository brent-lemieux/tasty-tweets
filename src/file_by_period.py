from to_csv import to_csv
from clean_tweets import clean_pipeline
from load_and_process import load_csv
import pandas as pd
import os

path = '/Users/blemieux/projects/tweets/byday'
csv_path = '/Users/blemieux/projects/tweets/ts_csv'

def load_periods(base_path):
    for day in os.listdir(base_path)[1:]:
        day_path = base_path + '/' + day
        tweets = clean_pipeline(day_path)
        date = '2017-' + day
        cs_dt = (date + ',') * len(tweets)
        dates = cs_dt.split(',')[:-1]
        df = pd.DataFrame({'tweets':tweets, 'date':dates})
        df.to_csv('{}/{}tweets.csv'.format(csv_path,date))


def combine_data(csv_path):
    init_df = pd.read_csv('{}/2017-01-27tweets.csv'.format(csv_path), index_col=0)
    for f in os.listdir(csv_path)[2:]:
        try:
            new = pd.read_csv('{}/{}'.format(csv_path, f), index_col=0)
            init_df = pd.concat([init_df, new])
        except:
            try:
                new = pd.read_excel('{}/{}'.format(csv_path, f))
                init_df = pd.concat([init_df, new])
            except:
                print 'Failed on:', f
                pass
    init_df.dropna(inplace=True)
    init_df['date'] = init_df['date'].map(lambda x: str(x)[:10])
    init_df.to_csv('/Users/blemieux/projects/tweets/csv/master.csv')
    print "Success!"
    return init_df


def process_master(read_f_path, wrie_f_path):
    df = load_csv(read_f_path, lemma=True, slang=True)
    df = df[['date', 'tweets']]
    df.to_csv(wrie_f_path)



if __name__ == '__main__':
    # load_periods(path)
    df = combine_data(csv_path)
    process_master('/Users/blemieux/projects/tweets/csv/master.csv', '/Users/blemieux/projects/tweets/csv/clean_masterf.csv')
