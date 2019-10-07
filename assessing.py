### Imports
import pandas as pd
import random
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind
from modelling import update_df_with_tags, main_modelling

np.random.seed(42)

### Constants
tag_column = 'tags'
tokens_column = 'text'
title_column = 'Title'
category_column = 'category'
train_test_split = 0.8
pvalue_threshold = 0.05


###Main function
def main_assessing(df, model_res, unique_tags_train, parameters):
    # Split train and test sets
    corpus_train = df.copy()
    corpus_test = split_train_test(corpus_train)

    # Used for robustness
    model_res_test, unique_tags_test = main_modelling(df=corpus_test,
                                                      parameters=parameters)
    # Get updated df
    corpus_train = update_df_with_tags(corpus_train, model_res)
    corpus_test = update_df_with_tags(corpus_test, model_res_test)

    # Assess the models
    # M1: Robustness
    corpus_evaluated, robustness_score = evaluate_robustness(corpus_train, corpus_test)

    # M2: Separation
    t_tests, separation_score = evaluate_separation(corpus_train, unique_tags_train)

    # M3: Relevance
    relevance_score = evaluate_relevance(corpus_train)

    return robustness_score, separation_score, relevance_score


### Core functions

# Method 1: Robustness
def split_train_test(df, train_test_split=0.8):
    '''
    Creates a test set which is a randomly selected fraction of all the tokens

    :param df: the preprocessed dataframe
    :param train_test_split: the percentage of tokens to randomly select
    :return: two dataframes: one with 100% of the tokens  and one with a percentage of the tokens
    '''
    df_test = df.copy()
    df_test[tokens_column] = df.apply(
        lambda x: random.sample(x[tokens_column], round(len(x[tokens_column]) * train_test_split)), axis=1)

    return df_test


def compute_common_tags(row):
    '''
    Computes the percentage of common tags between the train and the test sets for a specific document

    :param row: row of the aggregated dataframe with tags
    :return: percentage of common tags between train and test sets
    '''
    train_tags = row['tags']
    test_tags = row['tags_test']
    nb_total_tags = len(train_tags) + len(test_tags)
    nb_common_tags = len(set(train_tags) & set(test_tags))
    percentage_common_tags = 2 * (nb_common_tags / nb_total_tags)

    return percentage_common_tags


def evaluate_robustness(df_train, df_test):
    '''
    Compute the percentage of common tags between the train and the test sets for the entire corpus

    :param df_train: output of the model with the train dataset
    :param df_test: output of the model with the test dataset
    :return: dataframe with robustness score per document, mean score for the entire corpus
    '''
    df_train['tags_test'] = df_test.tags
    df_train['robustness_score'] = df_train.apply(compute_common_tags, axis=1)
    mean_robustness_score = df_train.robustness_score.mean()
    return df_train, mean_robustness_score


# Method 2: Separation
def evaluate_separation(df, unique_tags, pvalue_threshold=0.05):
    """
    Assess if the tag of a cluster is significantly more in the cluster than in the rest of the corpus

    :param tag_column: tag column name
    :param unique_tags: list of unique tags
    :param pvalue_threshold: threshold to reject the null hypothesis
    :return: binary value (1 if the mean of the tfidf of the one cluster in statistically different from the rest of the corpus)
    """
    tag_column = df['tags']
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,  # already tokenized
        preprocessor=lambda x: x,  # already tokenized
        token_pattern=None
    )
    feature_matrix = vectorizer.fit_transform(tag_column)
    tfidf = pd.DataFrame(feature_matrix.todense(),
                         columns=vectorizer.get_feature_names())

    tag_column = tag_column.apply(lambda x: str(x))

    t_test = {}
    for tag in unique_tags:
        # Compute the tfidf for the documents of interest
        tfidf_doi = np.array(tfidf[tag_column.str.contains(tag)][tag])
        # Compute the tfidf for the document of non-interest
        tfidf_doni = np.array(tfidf[~tag_column.str.contains(tag)][tag])
        # Compute t-test
        t_test_results = ttest_ind(tfidf_doi, tfidf_doni)
        t_test[tag] = (t_test_results.pvalue < pvalue_threshold) * 1

    # Compute the percentage of tags that reject the null hypothesis
    per_tags = sum(t_test.values()) / len(t_test.values())

    return t_test, per_tags


# Method 3: Relevance

def relevance_ratio(row):
    """
    For each row adds our relevance metric which matches the tags found with the bag of words from the title
        @row: Full row of the input dataframe
    """

    count = 0
    no_tags = len(row["Title"])
    partial_ratio_treshold = 80

    # for each tag check if the tag is present in the title
    for tag in row["tags"]:

        for word in row["Title"]:

            partial_ratio = fuzz.partial_ratio(tag, word)

            if partial_ratio > partial_ratio_treshold:
                count += 1
                break

    try:
        relevance_title = (count / no_tags)
    except:
        relevance_title = np.nan

    row["relevance_title"] = relevance_title

    return row


def add_relevance_column(df):
    """
    Evaluates the pertinence of our unsupervised labelling model
        @df: Input Dataframes
    """

    df = df.apply(relevance_ratio, axis=1)
    return df


def get_metrics(df):
    """
    Computes the mean score accross all rows
    """

    relevance_title_metric = df["relevance_title"].mean()

    return relevance_title_metric


def evaluate_relevance(df):
    """
    Evaluates the relevance of our model
    """

    df = add_relevance_column(df)
    relevance_title_metric = get_metrics(df)

    return relevance_title_metric
