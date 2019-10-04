#from scrapping import main_scrapping
from preprocessing import apply_preprocessing
from modelling import main_modelling
from assessing import main_assessing

import pandas as pd


#Constants
parameters = {"method_choice": ["kmeans_model", "kmeans_model", "kmeans_model", "kmeans_model", "kmeans_model"],
              "vector_choice": ["tfidf", "tfidf", "tfidf", "tfidf", "tfidf"],
              "tag_choice": ["centroid", "centroid", "centroid", "centroid", "centroid"]}


#Load the corpus
#corpus_scrapped = main_scrapping(apply_scrapping=False)

corpus_scrapped = pd.read_csv("./data/corpus.csv")

#Load the preprocessed data
corpus = apply_preprocessing(corpus_scrapped)


#Model
model_res, unique_tags_train = main_modelling(df=corpus,
                           max_depth=5,
                           parameters=parameters,
                           max_cluster_by_step=5,
                           min_size_of_a_cluster=11)


#Assessing
robustness_score, separation_score, relevance_evaluation = main_assessing(corpus, model_res, unique_tags_train)
