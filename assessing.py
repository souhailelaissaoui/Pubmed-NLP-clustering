### Imports
import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind
from gensim.models import KeyedVectors, Word2Vec



### Constants
tag_column = 'tags'
tokens_column = 'text'
title_column = 'Title'
category_column = 'category'
train_test_split = 0.8
pvalue_threshold  = 0.05






### Core functions

# Method 1: Robustness
def split_train_test(df, train_test_split=0.8):
    '''
    Randomly selects a percentage of tokens

    :param df: the preprocessed dataframe
    :param train_test_split: the percentage of tokens to randomly select
    :return: two dataframes: one with 100% of the tokens  and one with a percentage of the tokens
    '''
    df_test = df.copy()
    df_test[tokens_column] = df.apply(lambda x: random.sample(x[tokens_column],round(len(x[tokens_column]) * train_test_split)), axis=1)

    return df, df_test


# Method 2: Separation
def evaluate_separation(tag_column, unique_tags, pvalue_threshold=0.05):
    '''
    Assess if the tag of a cluster is significantly more in the cluster than in the rest of the corpus

    :param tag_column: tag column name
    :param unique_tags: list of unique tags
    :param pvalue_threshold: threshold to reject the null hypothesis
    :return: binary value (1 if the mean of the tfidf of the one cluster in statistically different from the rest of the corpus)
    '''

    # Compute tfidf
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(tag_column)
    tfidf = pd.DataFrame(feature_matrix.todense(), columns=vectorizer.get_feature_names())

    # For each label, perform a t-test
    t_test = {}
    for tag in unique_tags:
        # Compute the tfidf for the documents of interest
        tfidf_doi = np.array(tfidf[tag_column.str.contains(tag)][tag])
        tfidf_doni = np.array(tfidf[~tag_column.str.contains(tag)][tag])
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


### Execution flow
#M1:

df, df_test = split_train_test(df)

#M2:
#unique_tags = # todo: add Gautier function
evaluate_separation(tag_column, unique_tags)

#M3:
# model = # todo: add Adrien model w2v
evaluate_pertinence(df, model=model)

