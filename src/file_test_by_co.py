import pandas as pd

path = '/Users/blemieux/projects/tweets/csv/'


def separate_files(path, f, period_no):
    f_path = path + f
    df = pd.read_excel(f_path)
    sbux = df[df['tweets'].str.contains('starbucks')]
    cmg = df[df['tweets'].str.contains('chipotle')]
    mcd = df[df['tweets'].str.contains('mcdonalds')]
    cos = [sbux, cmg, mcd]
    co_names = ['starbucks', 'chipotle', 'mcdonalds']
    for i ,co in enumerate(cos):
        co.to_excel('{}{}_{}.xls'.format(path ,co_names[i], period_no), index=False)

if __name__ == '__main__':
    separate_files(path, 'test1.xls', 1)
    separate_files(path, 'test2.xls', 2)
