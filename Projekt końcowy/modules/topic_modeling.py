import pandas as pd
from bertopic import BERTopic

from modules.setup import FIRST_SUBREDDIT, SECOND_SUBREDDIT, RESULTS_PATH
from modules.clean_data import get_clean_csv


def get_data():
    first, second = get_clean_csv()
    total = pd.concat([first, second])
    return first, second, total


def get_topics(df, name: str, num_topics: int = 16):
    model = BERTopic(language="english", verbose=True, nr_topics=num_topics, calculate_probabilities=True)
    _, _ = model.fit_transform(df["text"])
    visualizations = get_visualization(model, name, num_topics=num_topics)
    return model, visualizations


def get_visualization(model, name: str, num_topics: int):
    figs = {
        'topics': model.visualize_topics(title=f'{name} Intertopic Distance map'),
        'barchart': model.visualize_barchart(top_n_topics=int(num_topics / 2), title=f'{name} Topic Word Scores'),
        'heatmap': model.visualize_heatmap(top_n_topics=num_topics, title=f'{name} Similarity Matrix'),
        'hierarchy': model.visualize_hierarchy(top_n_topics=num_topics,
                                               title=f'{name} Hierarchical Clustering Dendrogram')
    }
    for fig_name, fig in figs.items():
        fig.write_image(f"{RESULTS_PATH}/topics/images/{name}_{fig_name}.png")
        fig.write_html(f"{RESULTS_PATH}/topics/html/{name}_{fig_name}.html")
    return figs


def main(num_topics: int = 16):
    first, second, total = get_data()

    results = dict()
    results[FIRST_SUBREDDIT] = get_topics(first, FIRST_SUBREDDIT, num_topics=num_topics)
    results[SECOND_SUBREDDIT] = get_topics(second, SECOND_SUBREDDIT, num_topics=num_topics)
    results['Total'] = get_topics(total, 'total', num_topics=int(num_topics * 2))
    return results


if __name__ == '__main__':
    main()
