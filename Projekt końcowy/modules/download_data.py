import dotenv
import pandas as pd
import praw
import re
from tqdm.auto import tqdm

from modules.setup import FIRST_SUBREDDIT, SECOND_SUBREDDIT, DATA_PATH, AUTH_PATH


def get_reddit():
    auth = dotenv.dotenv_values(AUTH_PATH)
    reddit = praw.Reddit(
        client_id=auth['CLIENT_ID'],
        client_secret=auth['CLIENT_SECRET'],
        user_agent=auth['USER_AGENT'])
    return reddit


def is_read_only():
    reddit = get_reddit()
    return reddit.read_only


def download_comments(subreddit: str, posts=3, unwrap_comment_limit=5, omit_posts=None):
    reddit = get_reddit()

    total = 0
    for submission in reddit.subreddit(subreddit).top(limit=posts, time_filter='year'):
        if omit_posts is None or submission.id not in omit_posts:
            total += submission.num_comments

    comments = []

    pbar = tqdm(total=total)
    for submission in reddit.subreddit(subreddit).top(limit=posts, time_filter='year'):
        if omit_posts is None or submission.id not in omit_posts:
            submission.comments.replace_more(limit=unwrap_comment_limit)
            for comment in submission.comments.list():
                comments.append(comment.body)
                pbar.update(1)
    pbar.close()
    df = pd.DataFrame(comments, columns=['text'])
    return df, comments


def clean(text: str):
    tmp = text.strip()
    tmp = tmp.replace('\n', ' ')
    tmp = tmp.replace('\r', ' ')
    tmp = tmp.replace('\t', ' ')
    tmp = re.sub(r'\s+', ' ', tmp)
    return tmp


def get_csv():
    first = pd.read_csv(f'{DATA_PATH}/{FIRST_SUBREDDIT}.csv')
    second = pd.read_csv(f'{DATA_PATH}/{SECOND_SUBREDDIT}.csv')
    return first, second


def main(total_posts=10, unwrap_comment_limit=30):
    first, first_comments = download_comments(FIRST_SUBREDDIT, posts=total_posts,
                                              unwrap_comment_limit=unwrap_comment_limit)
    second, second_comments = download_comments(SECOND_SUBREDDIT, posts=total_posts,
                                                unwrap_comment_limit=unwrap_comment_limit)

    first['text'] = first['text'].apply(lambda x: clean(x))
    second['text'] = second['text'].apply(lambda x: clean(x))

    first.to_csv(f'{DATA_PATH}/{FIRST_SUBREDDIT}.csv', index=False)
    second.to_csv(f'{DATA_PATH}/{SECOND_SUBREDDIT}.csv', index=False)


if __name__ == '__main__':
    print(f'Read only: {is_read_only()}')
    main()
