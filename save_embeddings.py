import pandas as pd
import numpy as np
import csv

def clean_embeddings(chain_string):
    clean_array= chain_string.strip('[]')
    return np.fromstring(clean_array, sep=' ')

#read the csv file containing the sentences and their corresponding embeddings
df= pd.read_csv('fr_overview_sentences_with_embeddings.csv')
# Convert string representation of lists back to actual lists
df['embeddings_mean'] = df['embeddings_mean'].apply(clean_embeddings)

embeddings_matrix = np.stack(df['embeddings_mean'].values)


# Guardamos la matriz numérica separada por tabulaciones (\t)
np.savetxt('vectors.tsv', embeddings_matrix, delimiter='\t')

print("vectors.tsv guardado exitosamente.")

df['sentence_safe'] = df['sentence'].astype(str).str.replace('\t', ' ').str.replace('\n', ' ')
# Seleccionamos solo las columnas que quieres de etiquetas
columnas_metadata = ['sentence_safe','tipo', 'file_name']

# Guardamos en TSV. 
#poner index=False para que no guarde el número de fila
df[columnas_metadata].to_csv('metadata.tsv', sep='\t', index=False)

print("metadata.tsv guardado exitosamente.")