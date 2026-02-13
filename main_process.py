import pandas as pd 
import numpy as np
from closest_embeddings import ClosestEmbeddings
from tqdm import tqdm

def clean_embeddings(chain_string):
    clean_array= chain_string.strip('[]')
    return np.fromstring(clean_array, sep=' ')

#read the csv file containing the sentences and their corresponding embeddings
df= pd.read_csv('fr_overview_sentences_with_embeddings.csv')
# Convert string representation of lists back to actual lists
df['embeddings_mean'] = df['embeddings_mean'].apply(clean_embeddings)

df_req=df.loc[df['tipo'] == 'FR'].copy()
print(df_req.head())
df_overview=df.loc[df['tipo'] == 'Overview'].copy()
print(df_overview.head())


fr_embeddings=np.stack(df_req['embeddings_mean'].values)
overview_embeddings=np.stack(df_overview['embeddings_mean'].values)
print(fr_embeddings.shape)
print(overview_embeddings.shape)

searcher = ClosestEmbeddings(fr_embeddings)
closest_indices_list=[]

for element in tqdm(overview_embeddings, desc='Finding closest fr embeddings', total=len(overview_embeddings)):
    closest_indices, similarities = searcher.find_closest(element, top_k=5)
    closest_indices_list.append(closest_indices)

# Add the closest indices to the dataframe
df_overview['closest_fr_indices'] = closest_indices_list
print(df_overview.head())
#add the corresponding sentences as well
df_overview['closest_fr_sentences'] = df_overview['closest_fr_indices'].apply(lambda indices: [df_req.iloc[i]['sentence'] for i in indices])
#add the corresponding system of the closest fr sentences
df_overview['closest_fr_systems'] = df_overview['closest_fr_indices'].apply(lambda indices: [df_req.iloc[i]['file_name'] for i in indices])
# Save the updated dataframe to a new CSV file
df_overview.to_csv('overview_with_closest_fr_indices.csv', index=False)
