import pandas as pd
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.preprocessing import DenseTransformer
from tqdm.auto import tqdm

from modules.setup import FIRST_SUBREDDIT, SECOND_SUBREDDIT, RESULTS_PATH
from modules.download_data import get_csv


def get_data(partial=False):
    first, second = get_csv()

    first['from'] = FIRST_SUBREDDIT
    second['from'] = SECOND_SUBREDDIT

    if partial:
        first = first.head(1000)
        second = second.head(1000)

    total = pd.concat([first, second], ignore_index=True)
    total_array = total.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        total_array[:, 0], total_array[:, 1], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def create_pipeline(model, transformer=TfidfTransformer, transformer_name='tfidf'):
    return Pipeline([
        ('vect', TfidfVectorizer()),
        (transformer_name, transformer()),
        ('to_dense', DenseTransformer()),
        ('clf', model)
    ])


def train_model(model, X_train, y_train):
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    return pipeline


def test_model(model, X_test, y_test, label=FIRST_SUBREDDIT):
    scores = {
        f'accuracy': accuracy_score(y_test, model.predict(X_test)),
        f'precision_{FIRST_SUBREDDIT}': precision_score(y_test, model.predict(X_test), pos_label=label),
        f'recall_{FIRST_SUBREDDIT}': recall_score(y_test, model.predict(X_test), pos_label=label),
        f'f1-score_{FIRST_SUBREDDIT}': f1_score(y_test, model.predict(X_test), pos_label=label),
        f'precision_{SECOND_SUBREDDIT}': precision_score(y_test, model.predict(X_test), pos_label=label),
        f'recall_{SECOND_SUBREDDIT}': recall_score(y_test, model.predict(X_test), pos_label=label),
        f'f1-score_{SECOND_SUBREDDIT}': f1_score(y_test, model.predict(X_test), pos_label=label)
    }
    return scores


def write_csv(results: dict, filename=FIRST_SUBREDDIT, type='classification'):
    with open(f'{RESULTS_PATH}/classification/{type}-{filename}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'model', f'accuracy_{filename}', f'precision_{filename}', f'recall_{filename}',
                         f'f1-score_{filename}'])
        for name, result in results.items():
            writer.writerow([name, result[f'accuracy'], result[f'precision_{FIRST_SUBREDDIT}'],
                             result[f'recall_{FIRST_SUBREDDIT}'], result[f'f1-score_{FIRST_SUBREDDIT}']])


def train_classifiers(X_train, X_test, y_train, y_test):
    first_results = dict()
    second_results = dict()

    models = {
        'GaussianNB': GaussianNB(),
        'KNeighborsClassifier': KNeighborsClassifier(n_jobs=-1),
        'RandomForestClassifier': RandomForestClassifier(n_jobs=-1),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'MLPClassifier': MLPClassifier()
    }
    pbar = tqdm(models.items())
    for name, model in models.items():
        pbar.set_postfix_str(f'Training {name}')
        cls = train_model(model, X_train, y_train)
        pbar.set_postfix_str(f'Testing {name}')
        first_results[name] = test_model(cls, X_test, y_test, label=FIRST_SUBREDDIT)
        second_results[name] = test_model(cls, X_test, y_test, label=SECOND_SUBREDDIT)
        pbar.update(1)
    pbar.close()

    return first_results, second_results


def compare_methods(X_train, X_test, y_train, y_test):
    binary_classifier = Pipeline([
        ('vect', CountVectorizer(binary=True)),
        ('dense', DenseTransformer()),
        ('clf', GaussianNB()),
    ])

    logaritmic_classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('dense', DenseTransformer()),
        ('clf', GaussianNB())])

    tfidf_classifier = create_pipeline(GaussianNB(),
                                       transformer=TfidfTransformer,
                                       transformer_name='tfidf')

    classifiers = {'binary': binary_classifier,
                   'logaritmic': logaritmic_classifier,
                   'tfidf': tfidf_classifier}
    first_results = dict()
    second_results = dict()

    pbar = tqdm(classifiers.items())
    for name, classifier in classifiers.items():
        pbar.set_postfix_str(f'Training {name}')
        classifier.fit(X_train, y_train)
        pbar.set_postfix_str(f'Testing {name}')
        first_results[name] = test_model(classifier, X_test, y_test, label=FIRST_SUBREDDIT)
        second_results[name] = test_model(classifier, X_test, y_test, label=SECOND_SUBREDDIT)
        pbar.update(1)

    return first_results, second_results


def get_classification_df():
    method_f = pd.read_csv(f'{RESULTS_PATH}/classification/methods-{FIRST_SUBREDDIT}.csv')
    method_s = pd.read_csv(f'{RESULTS_PATH}/classification/methods-{SECOND_SUBREDDIT}.csv')
    classifier_f = pd.read_csv(f'{RESULTS_PATH}/classification/classifiers-{FIRST_SUBREDDIT}.csv')
    classifier_s = pd.read_csv(f'{RESULTS_PATH}/classification/classifiers-{SECOND_SUBREDDIT}.csv')

    return classifier_f, classifier_s, method_f, method_s


def main(partial=False):
    X_train, X_test, y_train, y_test = get_data(partial=partial)

    first_classifiers, second_classifiers = train_classifiers(X_train, X_test, y_train, y_test)
    first_methods, second_methods = compare_methods(X_train, X_test, y_train, y_test)

    return first_classifiers, second_classifiers, first_methods, second_methods


if __name__ == '__main__':
    cf, cs, mf, ms = main(partial=True)
    write_csv(cf, filename=FIRST_SUBREDDIT, type='classifiers')
    write_csv(cs, filename=SECOND_SUBREDDIT, type='classifiers')
    write_csv(mf, filename=FIRST_SUBREDDIT, type='methods')
    write_csv(ms, filename=SECOND_SUBREDDIT, type='methods')
