### Imports
import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind
from modelling import update_df_with_tags, main_modelling



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
                                                      max_depth=5,
                                                      parameters=parameters,
                                                      max_cluster_by_step=5,
                                                      min_size_of_a_cluster=11)
    # Get updated df
    corpus_train = update_df_with_tags(corpus_train, model_res)
    corpus_test = update_df_with_tags(corpus_test, model_res_test)

    # Assess the models
    # M1: Robustness
    corpus_evaluated, robustness_score = evaluate_robustness(corpus_train, corpus_test)

    # M2: Separation
    t_tests, separation_score = evaluate_separation(corpus_train, unique_tags_train)

    # M3:
    # model = # todo: add Adrien model w2v
    #relevance_evaluation = evaluate_relevance(corpus_train, model=model)

    # TODO - adapter les outputs de main_assessing
    return robustness_score, separation_score#, relevance_evaluation



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
    percentage_common_tags = 200 * (nb_common_tags / nb_total_tags)

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
    '''
    Assess if the tag of a cluster is significantly more in the cluster than in the rest of the corpus

    :param tag_column: tag column name
    :param unique_tags: list of unique tags
    :param pvalue_threshold: threshold to reject the null hypothesis
    :return: binary value (1 if the mean of the tfidf of the one cluster in statistically different from the rest of the corpus)
    '''
    tag_column = df['tags']
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,  # already tokenized
        preprocessor=lambda x: x,  # already tokenized
        max_features=500,
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
    per_tags = sum(t_test.values()) / len(t_test.values()) * 100

    return t_test, per_tags


# Method 3: Relevance

def relevance_ratio(row):
    '''
    Computes the relevance ratio for a title

    :param row: row of the preprocessed dataframe
    :return: same row with a relevance metric
    '''
    count = 0
    nb_words_title = len(row[title_column])

    for tag in row[tag_column]:
        if tag in row[title_column]:
            count += 1

    relevance_metric = count / nb_words_title
    row["relevance_metric"] = relevance_metric

    return row


def relevance_category(row, model):
    word_vectors = model.wv
    category = row[category_column]
    number_tags = len(row[tag_column])

    for tag in row[tag_column]:

        try:
            cum_score += word_vectors.n_similarity([tag], [category])
        except:
            print("category word does not appear in all texts combined")
            number_tags -= 1
            continue

    row['mean_score'] = cum_score / number_tags

    return row


def evaluate_relevance(df, model):
    df = df.apply(relevance_ratio, axis=1)
    df = df.apply(lambda row: relevance_category(row, model), axis=1)

    return df

