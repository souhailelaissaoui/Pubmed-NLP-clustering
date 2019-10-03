### Imports
import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind


### Constants
train_test_split = 0.8
tag_column = 'tags'
pvalue_threshold  = 0.05




### Core functions

# Method 1: Robustness
def get_robustness_evaluation(df, train_test_split=0.8):
    '''
    Create two dataframes: one with 100% of the tokens and other one with a percentage of the data
    '''

    # Compute
    df['test_tokens'] = df.apply(lambda x: random.sample(x[tokens_col],round(len(x[tokens_col]) * train_test_split)), axis=1)

    return df


# Method 2: Separation
def get_seperation_evaluation(tag_column, unique_tags, pvalue_threshold=0.05):
    '''
    Returns if the mean of the tfidf of the cluster with a label is statistically different from the group which has not this label
    (t-test)
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

    return t_test



# Method 3: Relevance



### Execution flow
#M1:
get_robustness_evaluation(df)

#M2:
unique_tags = # todo: add Gautier function
get_seperation_evaluation(tag_column, unique_tags)


