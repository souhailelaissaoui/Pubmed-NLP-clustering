from preprocessing import text_preprocessing, vectorize_corpus
from modelling import get_df_for_centroid, inception_clustering_model
from assessing import evaluate_relevance, evaluate_separation


#Load the preprocessed data
corpus = text_preprocessing()

#Vectorize the corpus
corpus = vectorize_corpus(corpus.sample(n=5), methods=["w2v", "tfidf"])

#Split train and test sets
corpus_train, corpus_test = split_train_test(corpus)


#Model
corpus_for_centroid = get_df_for_centroid(corpus)
model_res = inception_clustering_model(df=corpus,
                                       df_for_centroid=corpus_for_centroid,
                                       article_ID_list=corpus.article_ID.to_list(),
                                       max_depth=5,
                                       depth=0,
                                       parameters=parameters,
                                       max_cluster_by_step=5,
                                       min_size_of_a_cluster=11,
                                       local_node=MetaCluster([], ["Root"]))




#Assess the models
#M1:

#M2: separation
unique_tags = get_set_of_all_tags(model_res)
separation_evaluation = evaluate_separation(tag_column, unique_tags)

#M3:
# model = # todo: add Adrien model w2v
relevance_evaluation = evaluate_relevance(df, model=model)