#from scrapping import main_scrapping
from loading import main_loading
from preprocessing import main_preprocessing
from modelling import main_modelling
from assessing import main_assessing

#Constants
method = "kmeans_model"
vector = "tfidf"
tag_choice = "centroid"

parameters = {"method_choice": [method]*10,
              "vector_choice": [vector]*10,
              "tag_choice": [tag_choice]*10,
              "max_depth": 10,
              "max_cluster_by_step": 10,
              "min_size_of_a_cluster": 11}


#Load the corpus
main_loading(run_loading=True)


#Load the preprocessed data
corpus = main_preprocessing(run_preprocessing=True)


#Model
model_res, unique_tags_train = main_modelling(df=corpus, parameters=parameters)


#Assessing
robustness_score, separation_score, relevance_score = main_assessing(corpus, model_res, unique_tags_train, \
                                                                          parameters)

#Print results
print("parameters")
print(parameters)
print("robustness_score")
print(robustness_score)
print("separation_score")
print(separation_score)
print("relevance_score")
print(relevance_score)
print("Unique tags")
print(len(unique_tags_train))

