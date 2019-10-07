### Imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(42)
### Constants

column_words = 'text'
# TODO - P2
nb_tags = 1


### Main function
def main_modelling(df, parameters):
    """
    Main function that calls the recurcive function for the clustering model with the good entry point

    :param df: preprocessed data frame
    :param max_depth: maximal depth for the tree of clusters
    :param parameters: information to choose the vector, the model, and the tag at each depth
    :param max_cluster_by_step: max cluster by step for the kmean model
    :param min_size_of_a_cluster: minimal size at which the tree stops (leaf)
    :return: an MetaCluster object with the tree of the result
    """
    corpus_for_centroid = get_df_for_centroid(df)
    article_ID_list_racine = df.article_ID.to_list()
    col = df.tfidf_features[0]
    max_depth = parameters["max_depth"]
    max_cluster_by_step = parameters["max_cluster_by_step"]
    min_size_of_a_cluster = parameters["min_size_of_a_cluster"]
    model_res = inception_clustering_model(df,
                                           corpus_for_centroid,
                                           article_ID_list_racine,
                                           col,
                                           parameters=parameters,
                                           max_depth=max_depth,
                                           depth=0,
                                           max_cluster_by_step=max_cluster_by_step,
                                           min_size_of_a_cluster=min_size_of_a_cluster,
                                           local_node=MetaCluster([], ["Root"]))

    unique_tags = get_set_of_all_tags(model_res)
    return model_res, unique_tags


### Class definition

class MetaCluster(object):
    def __init__(self, index_list, tag):
        self.index_list = index_list
        self.tag = tag
        self.children_clusters = []

    def add_under_cluster(self, obj):
        self.children_clusters.append(obj)

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.tag) + ": " + str(len(self.index_list)) + " documents" + "\n"
        for cluster in self.children_clusters:
            ret += cluster.__repr__(level + 1)
        return ret


### Core functions
def get_df_for_centroid(df):
    """
    Get a structured data frame with one column per word to use the centroid method to get the tags
    :param df: output of the preprocessing part
    :return: a dataframe with a line per document and a column per word, with tfidf values
    """
    col = df.tfidf_features[0]
    list_words = []
    for i, j in df.tfidf.iteritems():
        list_words.append(j)
    X = np.array(list_words)
    df_for_centroid = pd.DataFrame(data=X, columns=col)
    return (df_for_centroid)


def inception_clustering_model(df,
                               df_for_centroid,
                               article_ID_list,
                               col,
                               parameters,
                               max_depth=3,
                               depth=0,
                               max_cluster_by_step=5,
                               min_size_of_a_cluster=10,
                               local_node=MetaCluster([], "Racine")):
    """
    Model creating the tree of clusters to define at each node a tag
    :param df: global preprocessed data frame
    :param df_for_centroid: cf function above, tfidf only dataframe
    :param article_ID_list: list of the relevant article_ID at the stage
    :param col: names of the words for the tfidf vector
    :param max_depth: maximum number of iteration (depth) in the tree, doesn't change
    :param depth: current depth at that stage, changes
    :param parameters: information about the relevant model / vectorization to use for kmeans
    :param max_cluster_by_step: maximal number of cluster under each node
    :param min_size_of_a_cluster: minimal size of a cluster to start a new kmean on it
    :param local_node: reference to the node for which the function is called
    :return: a MetaCluster class tree
    """
    # TODO latter - Testing parameters
    if len(parameters["method_choice"]) < max_depth:
        raise AttributeError("method_choice in method_parameter needs to be the same length as depth value")

    # Checking if it is not a leaf and then applying the method
    if (depth < max_depth) and (len(article_ID_list) >= min_size_of_a_cluster):
        # Getting the 2 methods used in the cluster part
        method_choice = parameters["method_choice"][depth]
        vector_choice = parameters["vector_choice"][depth]
        tag_choice = parameters["tag_choice"][depth]
        # Applying the clustering model and getting the optimal number of cluster, the article_IDs and the dataframe
        # index
        optimal_k, clusters_article_ID, df_index = cluster_model(df,
                                                                 method_choice,
                                                                 vector_choice,
                                                                 local_indexes=article_ID_list,
                                                                 k_max=max_cluster_by_step)
        # Looking for the best tag using the node list and the clusters lists, with the df index
        if tag_choice == "tfidf":
            tags = get_label_from_tfidf(df,
                                        article_ID_list,
                                        clusters_article_ID,
                                        df_index,
                                        nb_tags=nb_tags)
        elif tag_choice == "centroid":
            tags = get_label_from_centroid(df,
                                           df_for_centroid,
                                           article_ID_list,
                                           clusters_article_ID,
                                           col,
                                           df_index,
                                           depth)
        else:
            raise AttributeError("tag_choice in method_parameter not implemented")

        # For each cluster, store the information in a new node, add it to the local node, and
        # call the same function with the relevant parameters
        for new_cluster_number in range(0, optimal_k):
            best_tag = tags[new_cluster_number]
            new_node = MetaCluster(clusters_article_ID[new_cluster_number], best_tag)
            local_node.add_under_cluster(new_node)
            inception_clustering_model(df=df,
                                       df_for_centroid=df_for_centroid,
                                       article_ID_list=clusters_article_ID[new_cluster_number],
                                       col=col,
                                       parameters=parameters,
                                       max_depth=max_depth,
                                       depth=depth + 1,
                                       max_cluster_by_step=max_cluster_by_step,
                                       min_size_of_a_cluster=min_size_of_a_cluster,
                                       local_node=new_node)

    # After all the recursive function calls, we only return the root node
    if depth == 0:
        print(local_node)
        return (local_node)


def get_index_list(k_optim, labels, articles_ID_indexes):
    """
    Get the output of Kmeans model and convert it into list of IDs (article IDs or index of the df)
    :param k_optim: nb of clusters from Kmeans
    :param labels: list of the associated clusters for each document
    :param articles_ID_indexes: list of the article id (with same index as the df)
    :return: both article id list and df index as list of cluster lists
    """
    dataframe_index = [[i for i, cluster in enumerate(labels) if cluster == j] for j in np.arange(k_optim)]
    article_id_list = [[articles_ID_indexes[i] for i in j] for j in dataframe_index]
    return article_id_list, dataframe_index


def cluster_model(df, method_choice, vector_choice, local_indexes, k_max, k_min=2):
    """
    On one hand :
    kmeans model that thakes a array of vector (a vector is a document) from the recursive approach
    and tries the different kmeans models up to k_max clusters
    Returns indexes of the clusters and number of cluster chosen

    On the other hand :
    :param df: global df
    :param method_choice: kmeans or other
    :param vector_choice: tfidf or w2vec
    :param local_indexes: list of article_IDs to consider as a main node in which to find clusters
    :param k_max: maximal number of clusters
    :param k_min: minimal number of clusters
    :return: k_optim, article_id_list, dataframe_index for next clusters
    """
    # Kmean implementation
    if method_choice == "kmeans_model":
        silhouete_scores = []
        # Each row of the matrix is a document and each column a dimension of its vector
        vector_matrix = np.array(df.loc[df.article_ID.isin(local_indexes), vector_choice].to_list())
        # Finding the best number of cluster
        for k in np.arange(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k).fit(vector_matrix)
            labels = kmeans.labels_
            # To avoid errors with only one cluster (potentially removed with a cleaned dataset)
            if sum(labels) > 0:
                silhouete_scores.append(silhouette_score(vector_matrix, labels, metric='euclidean'))
            else:
                silhouete_scores.append(0)
        # If no k is relevant, all silhouete scores are null, then we skip that node
        if sum(silhouete_scores) > 0:
            k_optim = np.argmax(silhouete_scores) + k_min
            kmeans = KMeans(n_clusters=k_optim).fit(vector_matrix)
            labels = kmeans.labels_
            # to match article ids with kmean output
            article_ID_indexes = df.loc[df.article_ID.isin(local_indexes), :].article_ID.tolist()

            article_id_list, dataframe_index = get_index_list(k_optim, labels, article_ID_indexes)
        else:
            return -1, [], []

    elif method_choice == "dbscan_model":
        return ()

    else:
        raise AttributeError("Model choice is not available, check method parameters")

    return k_optim, article_id_list, dataframe_index


# TODO
def dbscan_model():
    return ()


def get_label_from_tfidf(df, list_node, list_clusters, df_index, nb_tags=nb_tags, column_words=column_words):
    """
    Selects the tokens from the corpus at the node.
    Computes the tfidf score for each token and document of the corpus within the node.
    Sums up the scores for each token from documents from the same cluster.
    And returns the tokens with the greatest aggregated tfidf for each cluster.
    :param df: global df
    :param list_node: article IDs of the father node
    :param list_clusters: clusters with their article ID
    :param df_index: clusters with their index in the df
    :param nb_tags: nb of tags per cluster in the tfidf method
    :param column_words: name of the column containing the whole text
    :return:
    """
    # Get the documents in the corpus of interest
    tokens_node = df.loc[df.article_ID.isin(list_node), column_words]

    # Create the tf-idf feature matrix
    # vectorizer = TfidfVectorizer()

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,  # already tokenized
        preprocessor=lambda x: x,  # already tokenized
        max_features=500,
        token_pattern=None
    )

    feature_matrix = vectorizer.fit_transform(tokens_node)

    # Get the words
    tfidf_names = vectorizer.get_feature_names()

    labels_clusters = []

    for cluster in df_index:
        tmp = feature_matrix[cluster, :]

        # Get the greatest aggregated tfidf
        tfidf_scores = tmp.mean(axis=0)
        # label_index = np.argsort(tfidf_scores)[-nb_tags:].item(nb_tags-1) #Problem TO DO
        label_index = tfidf_scores.argmax()

        # Get the word associated with the greated tfidf
        label = tfidf_names[label_index]
        labels_clusters.append([label])

    return labels_clusters


def get_label_from_centroid(df,
                            df_for_centroid,
                            list_node,
                            list_clusters,
                            col,
                            df_index,
                            depth):
    """
    Selects the tokens from the corpus at the node.
    Computes the vector mean.
    And returns the tokens with the greatest aggregated tfidf for each cluster.
    :param df: global df
    :param df_for_centroid: df with all tfidf
    :param list_node: article IDs of the father node
    :param list_clusters: clusters with their article ID
    :param col: names of the words in the tfidf column
    :param df_index: clusters with their index in the df
    :param depth: actual depth
    :return:
    """
    labels_clusters = []
    for cluster in df_index:
        tmp = df_for_centroid.loc[cluster, :]

        # Get the greatest aggregated tfidf
        tfidf_scores = tmp.mean(axis=0)
        # label = tfidf_scores.argmax()
        label = [col[i] for i in np.argsort(-tfidf_scores)[0:(depth + 1)].to_list()]
        labels_clusters.append(label)

    return labels_clusters


def get_set_of_all_tags(tree_result):
    """
    Return the list of all the tags of a tree
    :param tree_result: result of the main architecture function
    :return: the whole list of tags (unique)
    """
    """
    Return the list of all the tags of a tree
    """
    res = []

    def add_tag(local_node):
        for local_tag in local_node.tag:
            res.append(local_tag)
        for child in local_node.children_clusters:
            add_tag(child)

    for first_child in tree_result.children_clusters:
        add_tag(first_child)
    return set(res)


def update_df_with_tags(df, tree_result):
    size_df = len(df)
    tags = [[]] * size_df

    def add_tag_to_list(node):
        local_tag_list = node.tag
        local_list = node.index_list
        for local_article_ID in local_list:
            list_id = df.loc[df.article_ID == local_article_ID].index.tolist()[0]
            tags[list_id] = tags[list_id] + local_tag_list
        for children in node.children_clusters:
            add_tag_to_list(children)

    add_tag_to_list(tree_result)

    df["tags"] = tags

    for i in df.index:
        df.at[i, "tags"] = list(set(list(df.loc[i, "tags"])))
    return (df)
