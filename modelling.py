### Imports

## Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


### Constants

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
    list_words = []
    for i, j in df.tfidf.iteritems():
        list_words.append(j)
    X = np.array(list_words)
    df_for_centroid = pd.DataFrame(data=X, columns=col)
    return (df_for_centroid)


def inception_clustering_model(df,
                               df_for_centroid,
                               article_ID_list,
                               max_depth=3,
                               depth=0,
                               parameters=parameters,
                               max_cluster_by_step=5,
                               min_size_of_a_cluster=10,
                               local_node=MetaCluster([], "Racine")):
    """
    Model creating the tree of clusters to define at each node a tag
    :param df: global preprocessed data frame
    :param df_for_centroid: cf function above, tfidf only dataframe
    :param article_ID_list: list of the relevant article_ID at the stage
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
                                        df_index)
        elif tag_choice == "centroid":
            tags = get_label_from_centroid(df,
                                           df_for_centroid,
                                           article_ID_list,
                                           clusters_article_ID,
                                           df_index,
                                           depth)
        else :
            raise AttributeError("tag_choice in method_parameter not implemented")

        # For each cluster, store the information in a new node, add it to the local node, and
        # call the same function with the relevant parameters
        for new_cluster_number in range(0, optimal_k):
            best_tag = tags[new_cluster_number]
            new_node = MetaCluster(clusters_article_ID[new_cluster_number], best_tag)
            local_node.add_under_cluster(new_node)
            print("--------------------------------")
            print("Profondeur : " + str(depth))
            print("Cluster : " + str(best_tag))
            print("Nb documents : " + str(len(clusters_article_ID[new_cluster_number])))
            inception_clustering_model(df=df,
                                       article_ID_list=clusters_article_ID[new_cluster_number],
                                       max_depth=max_depth,
                                       depth=depth + 1,
                                       parameters=parameters,
                                       max_cluster_by_step=max_cluster_by_step,
                                       min_size_of_a_cluster=min_size_of_a_cluster,
                                       local_node=new_node)

    # After all the recursive function calls, we only return the root node
    if depth == 0:
        return (local_node)

### Execution flow (to be moved to main)


## Data
# TODO
df = pd.read_json("corpus4.json")
col = df.tfidf_features[0]
df_for_centroid = get_df_for_centroid(df)
