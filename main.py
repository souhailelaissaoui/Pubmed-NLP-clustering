from preprocessing import text_preprocessing, vectorize_corpus
from modelling import main_modelling
from assessing import evaluate_relevance, evaluate_separation, evaluate_robustness


#Load the preprocessed data
corpus = text_preprocessing()

#Vectorize the corpus
corpus = vectorize_corpus(corpus.sample(n=5), methods=["w2v", "tfidf"])

#Split train and test sets
corpus_train, corpus_test = split_train_test(corpus)


#Model
model_res = main_modelling(df=corpus_train,
                           max_depth=5,
                           parameters=parameters,
                           max_cluster_by_step=5,
                           min_size_of_a_cluster=11)


model_res_test = main_modelling(df=corpus_test,
                           max_depth=5,
                           parameters=parameters,
                           max_cluster_by_step=5,
                           min_size_of_a_cluster=11)

#Assess the models
#M1: Robustness
corpus_evaluated, robustness_score = evaluate_robustness(corpus_train, corpus_test)

#M2: Separation
unique_tags = get_set_of_all_tags(model_res)
separation_evaluation = evaluate_separation(tag_column, unique_tags)

#M3:
# model = # todo: add Adrien model w2v
relevance_evaluation = evaluate_relevance(df, model=model)