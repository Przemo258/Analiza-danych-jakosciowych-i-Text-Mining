import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from tqdm.auto import tqdm

from modules.setup import FIRST_SUBREDDIT, SECOND_SUBREDDIT, RESULTS_PATH
from modules.clean_data import get_clean_csv


def prepare():
    first, second = get_clean_csv()

    first_words = first['text'].sum().split()
    second_words = second['text'].sum().split()
    all_words = first_words + second_words

    only_first_words = [word for word in tqdm(first_words) if word not in second_words]
    only_second_words = [word for word in tqdm(second_words) if word not in first_words]

    words = {
        'first_words': first_words,
        'second_words': second_words,
        'all_words': all_words,
        'only_first_words': only_first_words,
        'only_second_words': only_second_words
    }
    return words


def plot(arr: list[str], title: str, filename: str = None):
    top_20 = Counter(arr).most_common(20)
    plt.bar([x[0] for x in top_20], [x[1] for x in top_20])
    plt.rcParams["figure.figsize"] = (8, 10)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.savefig(filename)
    plt.show()


def create_plots(lists: dict[str, list[str]]):
    first_words = lists['first_words']
    second_words = lists['second_words']
    all_words = lists['all_words']
    only_first_words = lists['only_first_words']
    only_second_words = lists['only_second_words']

    plot(first_words, f'{FIRST_SUBREDDIT} words', f'{RESULTS_PATH}/plots/{FIRST_SUBREDDIT}_words.png')
    plot(second_words, f'{SECOND_SUBREDDIT} words', f'{RESULTS_PATH}/plots/{SECOND_SUBREDDIT}_words.png')
    plot(all_words, 'All words', f'{RESULTS_PATH}/plots/All_words.png')

    plot(only_first_words, f'Only {FIRST_SUBREDDIT} words',
         f'{RESULTS_PATH}/plots/Only_{FIRST_SUBREDDIT}_words.png')
    plot(only_second_words, f'Only {SECOND_SUBREDDIT} words',
         f'{RESULTS_PATH}/plots/Only_{SECOND_SUBREDDIT}_words.png')


def wordcloud(arr: list[str], title: str, filename: str = None):
    text = " ".join(arr)
    wc = WordCloud(width=800, height=800,
                   background_color='white',
                   stopwords=STOPWORDS,
                   min_font_size=10,
                   scale=1.25).generate(text)
    wc.to_file(filename)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title)
    plt.imshow(wc)
    plt.show()


def create_wordclouds(lists: dict[str, list[str]]):
    first_words = lists['first_words']
    second_words = lists['second_words']
    all_words = lists['all_words']
    only_first_words = lists['only_first_words']
    only_second_words = lists['only_second_words']

    wordcloud(first_words, f'{FIRST_SUBREDDIT} words', f'{RESULTS_PATH}/wordclouds/{FIRST_SUBREDDIT}_words.png')
    wordcloud(second_words, f'{SECOND_SUBREDDIT} words', f'{RESULTS_PATH}/wordclouds/{SECOND_SUBREDDIT}_words.png')
    wordcloud(all_words, 'All words', f'{RESULTS_PATH}/wordclouds/All_words.png')

    wordcloud(only_first_words, f'Only {FIRST_SUBREDDIT} words',
              f'{RESULTS_PATH}/wordclouds/Only_{FIRST_SUBREDDIT}_words.png')
    wordcloud(only_second_words, f'Only {SECOND_SUBREDDIT} words',
              f'{RESULTS_PATH}/wordclouds/Only_{SECOND_SUBREDDIT}_words.png')


def main():
    prepared = prepare()
    create_wordclouds(prepared)
    create_plots(prepared)


if __name__ == '__main__':
    main()
