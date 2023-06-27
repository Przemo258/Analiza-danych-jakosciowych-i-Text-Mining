import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline

from modules.setup import FIRST_SUBREDDIT, SECOND_SUBREDDIT, RESULTS_PATH
from modules.clean_data import get_partial_clean_csv

sentiment = pipeline('sentiment-analysis', model='finiteautomata/bertweet-base-sentiment-analysis')
tqdm.pandas()


def get_data():
    first, second = get_partial_clean_csv()
    total = pd.concat([first, second], axis=0)
    return first, second, total


def get_stats(df: pd.DataFrame, name: str):
    values = df['label'].value_counts()
    count = df['label'].count()
    percentages = (values / count) * 100

    tmp = pd.concat([values, percentages], axis=1)
    tmp.columns = [f'{name}_count', f'{name}_percentage']
    tmp.loc[f'Total'] = [count, 100]
    return tmp


def sentiment_stats(df: pd.DataFrame, name: str):
    tmp = df['text'].progress_apply(lambda x: sentiment(x[:128])[0])
    tmp = tmp.apply(pd.Series)
    stats = get_stats(tmp, name)
    return stats, tmp


def get_sentiment_df():
    return pd.read_csv(f'{RESULTS_PATH}/sentiment.csv', index_col=0)


def main():
    first, second, total = get_data()

    first_stats, first_df = sentiment_stats(first, f'{FIRST_SUBREDDIT}')
    second_stats, second_df = sentiment_stats(second, f'{SECOND_SUBREDDIT}')

    total_df = pd.concat([first_df, second_df], axis=0)
    total_stats = get_stats(total_df, 'Total')

    result = pd.concat([first_stats, second_stats, total_stats], axis=1)
    result.to_csv(f'{RESULTS_PATH}/sentiment.csv', index=True)
    return result


if __name__ == '__main__':
    main()
