import pandas as pd

from modules.setup import FIRST_SUBREDDIT, SECOND_SUBREDDIT, RESULTS_PATH
from modules.download_data import get_csv
from modules.clean_data import get_clean_csv


def get_data():
    first, second = get_csv()
    first_clean, second_clean = get_clean_csv()

    total = pd.concat([first, second], axis=0)
    total_clean = pd.concat([first_clean, second_clean], axis=0)
    return first, first_clean, second, second_clean, total, total_clean


def summarize(df: pd.Series, name: str):
    tmp = df.describe()
    tmp.columns = [name]
    tmp.loc['length'] = df['text'].str.len().sum()
    tmp.loc['mean'] = df['text'].str.len().mean()
    tmp.loc['std'] = df['text'].str.len().std()
    return tmp


def main():
    first, first_clean, second, second_clean, total, total_clean = get_data()

    names = [FIRST_SUBREDDIT, f'{FIRST_SUBREDDIT} Clean', SECOND_SUBREDDIT, f'{SECOND_SUBREDDIT} Clean',
             'Total', 'Total Clean']
    results = []

    for df, name in zip([first, first_clean, second, second_clean, total, total_clean], names):
        results.append(summarize(df, name))

    result = pd.concat(results, axis=1)
    result.to_csv(f'{RESULTS_PATH}/stats.csv', index=True)
    return result


def get_stats_df():
    return pd.read_csv(f'{RESULTS_PATH}/stats.csv', index_col=0)


if __name__ == '__main__':
    results = main()
    print(results)
