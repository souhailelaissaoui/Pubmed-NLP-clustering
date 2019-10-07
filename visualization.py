### Imports
import pandas as pd
import numpy as np

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
    tableau_df = pd.DataFrame(columns=column_list)
    max_tags_tempo = [0] * max_depth

    def fill_df_recurcive(local_node, tempo_node_names, tempo_node_numbers, depth):
        global tableau_df
        try:
            # call function on children
            children = local_node.children_clusters
            for child_number in range(len(children)):
                max_tags_tempo[depth] += 1
                fill_df_recurcive(children[child_number],
                                  tempo_node_names.append(children[child_number].tag),
                                  tempo_node_numbers.append(max_tags_tempo[depth]),
                                  depth + 1)
        except:
            # add leaf
            information_list = tempo_node_names + [str(local_node.index_list())] + tempo_node_numbers
            added_row = pd.Series(information_list, index=column_list)
            tableau_df = pd.concat(tableau_df, added_row.to_frame().T)
            return ()

    fill_df_recurcive(model_res, [], [], 0)
    return tableau_df
