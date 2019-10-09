### Imports
import pandas as pd
import numpy as np
import csv
import copy


### Main function
def main_visualization(model_res, parameters):
    """

    :param model_res:
    :param parameters:
    :return:
    """
    # Get parameters
    max_depth = parameters["max_depth"]
    # Create final dataframe (empty for now)
    column_list = ["Node_name_" + str(i) for i in np.arange(1, max_depth + 1)] + \
                  ["Article_ID_List"] + \
                  ["Node_number_" + str(i) for i in np.arange(1, max_depth + 1)]
    tableau_list = []
    tempo_node_names = [""] * max_depth
    tempo_node_numbers = [""] * max_depth
    max_tags_tempo = [0] * max_depth

    def fill_df_recurcive(spamwriter, tempo_node_names, tempo_node_numbers, local_node, depth):
        if len(local_node.children_clusters) > 0:
            # call function on children
            children = local_node.children_clusters
            for child_number in range(len(children)):
                max_tags_tempo[depth] += 1
                tempo_node_names = copy.copy(tempo_node_names)
                tempo_node_names[depth] = children[child_number].tag
                tempo_node_numbers = copy.copy(tempo_node_numbers)
                tempo_node_numbers[depth] = max_tags_tempo[depth]
                fill_df_recurcive(spamwriter, tempo_node_names, tempo_node_numbers,
                                  children[child_number],
                                  depth + 1)
        else:
            # add leaf
            information_list = tempo_node_names + [str(local_node.index_list)] + tempo_node_numbers
            spamwriter.writerow(information_list)
            return ()

    with open('tableau_res.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';')
        spamwriter.writerow(column_list)
        fill_df_recurcive(spamwriter, tempo_node_names, tempo_node_numbers, model_res, 0)

    tableau_df_res = pd.read_csv("tableau_res.csv")

    return tableau_df_res
