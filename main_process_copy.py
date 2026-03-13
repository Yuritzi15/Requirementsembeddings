import pandas as pd 
import numpy as np
from closest_embeddings import ClosestEmbeddings
from tqdm import tqdm

def clean_embeddings(chain_string):
    """
    Convierte la representación en string de una lista a un array de numpy.
    Maneja casos donde el valor podría no ser string.
    """
    if isinstance(chain_string, str):
        clean_array = chain_string.strip('[]')
        # Verifica si la cadena está vacía después de limpiar
        if not clean_array:
            return np.array([]) 
        return np.fromstring(clean_array, sep=' ')
    return np.array([]) # Retorna array vacío si no es string válida

def get_min_max(df, embedding_column):
    """get the min and max distances of all the distances between the centroids and the fr sentences, to use them in the bins of the labels."""
    df_work = df.copy()
    df_work[embedding_column] = df_work[embedding_column].apply(clean_embeddings)

    #get only the rows with type Overview
    df_overview= df_work.loc[df_work['tipo'] == 'Overview'].copy()
    #get only the rows with type FR
    df_req = df_work.loc[df_work['tipo'] == 'FR'].copy().reset_index(drop=True)

    # Knowledge base
    fr_embeddings = np.stack(df_req[embedding_column].values)
    searcher = ClosestEmbeddings(fr_embeddings)

    all_distances = []
    for system_name, group in tqdm(df_overview.groupby('file_name'), desc=f"Calculating min and max distances of {embedding_column}"):
        system_embeddings = np.stack(group[embedding_column].values)
        centroid = np.mean(system_embeddings, axis=0)
        distances = searcher.euclidean_dist(centroid)
        all_distances.extend(distances)

    min_distance = min(all_distances)
    max_distance = max(all_distances)
    #print(f"Min distance: {min_distance}, Max distance: {max_distance}")
    return min_distance, max_distance, all_distances



def embeddings_process(df, embedding_column, output_name, labels,bins, column_system, text_column, distance_metric, min_distance, max_distance):
    """Process of cleaning, centroids compute and closest embeddings search.
    Args:
        df (pd.DataFrame): DataFrame with the data to process.
        embedding_column (str): Name of the column with the embeddings in string format.
        output_name (str): Name of the output CSV file.
        labels (list): List of labels for similarity bins.
        bins (list): List of bin edges for similarity labeling.
        """
    print("Column: ", embedding_column)

    #1 clean embeddings and separate by type
    print("Cleaning embeddings...")
    df_work = df.copy()
    df_work[embedding_column] = df_work[embedding_column].apply(clean_embeddings)

    #get only the rows with type Overview
    df_overview= df_work.loc[df_work['tipo'] == 'Overview'].copy()
    #get only the rows with type FR
    df_req = df_work.loc[df_work['tipo'] == 'FR'].copy().reset_index(drop=True)


    #2 Knowledge base
    print("Preparing knowledge base...")
    fr_embeddings = np.stack(df_req[embedding_column].values)
    searcher = ClosestEmbeddings(fr_embeddings)

    #3 Process by system
    systems_data = [] # to get the data of each system

    print("Grouping systems and calculating centroids...")
    for system_name, group in tqdm(df_overview.groupby(column_system), desc=f"Processing Systems of {embedding_column}"):

        #1 Centroid
        system_embeddings = np.stack(group[embedding_column].values)
        centroid = np.mean(system_embeddings, axis=0)

        #2 Concatenate text
        overview_sentences = group[text_column].astype(str).tolist()
        complete_text = ' '.join(overview_sentences)

        #3 Get fr distances and label them
        if distance_metric == 'cosine':
            res_labels, similarities = searcher.find_closest_cosine(centroid, labels, bins)
            print(similarities.min(), similarities.max())  # Debug: Print min and max similarity values to check the range

        elif distance_metric == 'euclidean':
            distances = searcher.euclidean_dist(centroid)
            
        bins=np.linspace(min_distance, max_distance, num=len(labels)+1)  # Create bins based on the min and max distances
        print("Bins: ", bins)  # Debug: Print the bins to check their values
        current_labels = pd.cut(distances, bins=bins, labels=labels, include_lowest=True)
        print(current_labels.value_counts())  # Debug: Print the count of each label to check the distribution
        searcher.histogramsglobal(distances, labels, system_name, min_distance, max_distance)
        """CSV BY SYSTEM ( ALL THE SYSTEMS HAVE THE SAME BINS)"""
        for i, row_req in df_req.iterrows():
            systems_data.append({
                'system': system_name,
                    'overview_text': complete_text,
                    'requrement_sentence': row_req[text_column],
                    'distance_label': current_labels[i],
                    'requirement_document': row_req['file_name'],
                    'similarity_value': distances[i]
                })

        # #4 Filter only "Alta" labels
        # closest_indices = [i for i, label in enumerate(res_labels) if label == 'Alta']        
        # print(f"System: {system_name} - Found {len(closest_indices)} FRs with 'Alta' similarity.")

        # #4 Save all in a temporary structure
        # systems_data.append({
        #     'system': system_name,
        #     'overview_text': complete_text,
        #     'overview_centroid': centroid, 
        #     'closest_fr_indices': closest_indices,
        #     'min_similarity': similarities.min(),
        #     'max_similarity': similarities.max()
        # })

    # # aux function to get info of fr sentences and systems by indices
    # def get_fr_info(indices, col_name):
    #     # iloc for reset index, loc for original index
    #     return [df_req.iloc[i][col_name] for i in indices]

    # #4 Build results DataFrame
    results_df = pd.DataFrame(systems_data)

    
    

    # # build the columns with the sentences and systems of the closest fr
    # results_df['closest_fr_sentences'] = results_df['closest_fr_indices'].apply(lambda idx: get_fr_info(idx, 'sentence'))
    # results_df['closest_fr_systems'] = results_df['closest_fr_indices'].apply(lambda idx: get_fr_info(idx, 'file_name'))

    # # Save results
    results_df.to_csv(output_name, index=False)
    print(f"Process finished. File saved in: {output_name}")

    #get the avg bin metric to measure the difference for each system
    results_df['distance_label']=results_df['distance_label'].astype(float)
    
    #filter 
    own_req=results_df[results_df['system']==results_df['requirement_document']]

    #calculate the bin
    stats = (
    own_req
    .groupby('system')['distance_label']
    .agg(['mean','min','max','count'])
    .reset_index()
)
    print(stats)
    stats.to_csv("average_bins_per_system_ex1.csv", index=False)


def main():
    "This version use the min and max of all the distances to create the bins, so all the systems share the same bins."
    # Read CSV
    df = pd.read_csv('filtered.csv')
    # df = pd.read_csv('fr_overview_sentences_with_embeddings_cls.csv')

    # Share configuration 
    labels = ['1', '2','3', '4', '5', '6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'] 
    bins = [0.0, 0.33, 0.66, 1.0] 
    column_system = 'file_name' 
    text_column = 'sentence'
    # distance_metric = 'cosine'
    distance_metric = 'euclidean'

    # Process for the column with BERT4SE embeddings
    embedding_column = 'embeddings_mean'
    output_name = 'ov_reqfr_gen_minmax_BERT_mean_filtered.csv'
    min_distance, max_distance, all_distances=get_min_max(df, embedding_column)
    print("Min distance: ", min_distance)
    print("Max distance: ", max_distance)
    #get the distribition of all the distances to see if the bins are well defined
    # searcher = ClosestEmbeddings(df[embedding_column].tolist())
    # searcher.histogram_distances(all_distances)



    embeddings_process(df, 
                       embedding_column, 
                       output_name, labels, 
                       bins, column_system, 
                       text_column,
                       distance_metric, 
                       min_distance,
                       max_distance
    )

    # #process for Vanilla BERT embeddings
    # embedding_column = 'embeddings'
    # output_name = 'ov_reqfr_same_minmax_BERT_cls.csv'
    # embeddings_process(df, 
    #                    embedding_column, 
    #                    output_name, labels, 
    #                    bins, column_system, 
    #                    text_column,
    #                    distance_metric
    # )

if __name__ == "__main__":
    main()