import re
import emoji
import numpy as np
import pandas as pd
import spacy
from tqdm.auto import tqdm

from modules.setup import FIRST_SUBREDDIT, SECOND_SUBREDDIT, DATA_PATH
from modules.download_data import get_csv

nlp = spacy.load('en_core_web_md')
tqdm.pandas()


def get_data():
    first, second = get_csv()
    return first, second


def clean_data(text: str):
    cleaned = []
    if text in ['[deleted]', '[removed]']:
        return np.nan
    tmp = emoji.demojize(text, delimiters=("", ""))
    tmp = nlp(tmp)
    for token in tmp:
        if token.is_stop or token.is_punct or token.is_space \
                or token.like_num or token.like_url or token.like_email:
            continue
        cleaned.append(token.lemma_)
    res = " ".join(cleaned)
    res = re.sub(r'[^a-zA-Z0-9 ]', ' ', res)
    res = re.sub(r'""', ' ', res)
    res = re.sub(r'\s+', ' ', res)
    res = res.strip()
    return res if len(res) >= 3 else np.nan


def partial_clean(text: str):
    if text in ['[deleted]', '[removed]']:
        return np.nan
    res = emoji.demojize(text, delimiters=("", ""))
    res = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', ' ', res)
    # res = re.sub(r'(https:\/\/)*(www.[\w./?=&]+)', '', res)
    res = re.sub(r'[^a-zA-Z0-9\' ]', ' ', res)
    res = re.sub(r'""', ' ', res)
    res = re.sub(r'\s+', ' ', res)
    res = res.strip()
    return res if len(res) >= 3 else np.nan


def get_clean_csv():
    first_df = pd.read_csv(f'{DATA_PATH}/{FIRST_SUBREDDIT}_clean.csv')
    second_df = pd.read_csv(f'{DATA_PATH}/{SECOND_SUBREDDIT}_clean.csv')
    return first_df, second_df


def get_partial_clean_csv():
    first_partial_df = pd.read_csv(f'{DATA_PATH}/{FIRST_SUBREDDIT}_partial_clean.csv')
    second_partial_df = pd.read_csv(f'{DATA_PATH}/{SECOND_SUBREDDIT}_partial_clean.csv')
    return first_partial_df, second_partial_df


def main():
    first, second = get_data()

    first_clean = first['text'].progress_apply(clean_data)
    second_clean = second['text'].progress_apply(clean_data)

    first_partial = first['text'].progress_apply(partial_clean)
    second_partial = second['text'].progress_apply(partial_clean)

    blacklist = ['remindme day']
    first_clean = first_clean[~first_clean.isin(blacklist)]
    second_clean = second_clean[~second_clean.isin(blacklist)]

    first_partial = first_partial[~first_partial.isin(blacklist)]
    second_partial = second_partial[~second_partial.isin(blacklist)]

    first_clean.dropna(inplace=True)
    first_clean.reset_index(drop=True, inplace=True)

    second_clean.dropna(inplace=True)
    second_clean.reset_index(drop=True, inplace=True)

    first_partial.dropna(inplace=True)
    first_partial.reset_index(drop=True, inplace=True)

    second_partial.dropna(inplace=True)
    second_partial.reset_index(drop=True, inplace=True)

    first_clean.to_csv(f'{DATA_PATH}/{FIRST_SUBREDDIT}_clean.csv', index=False)
    second_clean.to_csv(f'{DATA_PATH}/{SECOND_SUBREDDIT}_clean.csv', index=False)

    first_partial.to_csv(f'{DATA_PATH}/{FIRST_SUBREDDIT}_partial_clean.csv', index=False)
    second_partial.to_csv(f'{DATA_PATH}/{SECOND_SUBREDDIT}_partial_clean.csv', index=False)
    return first, first_partial, first_clean, second, second_partial, second_clean


if __name__ == '__main__':
    main()
