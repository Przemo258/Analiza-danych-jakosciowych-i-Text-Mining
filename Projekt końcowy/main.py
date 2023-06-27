import warnings

warnings.filterwarnings("ignore")

import time
import modules.setup as setup
import modules.download_data as dd
import modules.clean_data as cd
import modules.simple_stats as ss
import modules.wordclouds as wc
import modules.sentiment_analisys as sa
import modules.topic_modeling as tm
import modules.classification as cl


def sleepy_print(text):
    time.sleep(0.5)
    print('-' * 100)
    print(text)
    time.sleep(0.5)


def process():
    sleepy_print('Cleaning data')
    cd.main()
    sleepy_print('Calculating statistics')
    ss.main()
    sleepy_print('Creating wordclouds')
    wc.main()
    sleepy_print('Creating sentiment analysis')
    sa.main()
    sleepy_print('Getting topics')
    tm.main()
    sleepy_print('Creating classifiers')
    cl.main()
    sleepy_print('Done')


def no_download():
    print('Ignoring download')
    print(f'Using already downloaded data from {setup.FIRST_SUBREDDIT} and {setup.SECOND_SUBREDDIT}')
    process()


def main():
    print(f'Downloading data from {setup.FIRST_SUBREDDIT} and {setup.SECOND_SUBREDDIT}')
    print('This may take a while')
    dd.main(total_posts=setup.POSTS_TO_DOWNLOAD, unwrap_comment_limit=setup.UNWRAP_COMMENT_LIMIT)
    process()


if __name__ == "__main__":
    if input('Download data? (y/n): ').lower() == 'y':
        main()
    else:
        no_download()
