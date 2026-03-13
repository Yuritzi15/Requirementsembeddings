import pandas as pd
import numpy as np
import csv

def clean_embeddings(chain_string):
    if isinstance(chain_string, str):
        clean_array = chain_string.strip('[]')
        return np.fromstring(clean_array, sep=' ')
    return chain_string

# 1. Cargar datos
print("Cargando dataset...")
df = pd.read_csv('fr_overview_se_sentences_with_embeddings.csv')

# Convertir embeddings de string a numpy array
df['embeddings_mean_se'] = df['embeddings_mean_se'].apply(clean_embeddings)

# --- PASO 1: Preparar los Requerimientos (FR) ---
# Filtramos solo los FR
df_fr = df.loc[df['tipo'] == 'FR'].copy()
# Limpiamos el texto para evitar errores en el TSV (quitar tabs y saltos de linea)
df_fr['sentence_safe'] = df_fr['sentence'].astype(str).str.replace('\t', ' ').str.replace('\n', ' ')
df_fr['visual_label'] = 'Requirement' # Etiqueta para colorear en el visualizador

print(f"Requerimientos procesados: {len(df_fr)}")

# --- PASO 2: Calcular y Preparar los Centroides (Overview) ---
df_overview = df.loc[df['tipo'] == 'Overview']

centroid_vectors = []
centroid_metadata = []

# Agrupamos por sistema para calcular el promedio
for system_name, group in df_overview.groupby('file_name'):
    # 1. Calcular vector promedio (Centroide)
    matrix = np.stack(group['embeddings_mean_se'].values)
    centroid = np.mean(matrix, axis=0)
    
    # 2. Guardar vector
    centroid_vectors.append(centroid)
    
    # 3. Crear metadatos "artificiales" para el centroide
    centroid_metadata.append({
        'sentence_safe': f"CENTROID: {system_name}", # Texto que aparecerá al pasar el mouse
        'tipo': 'Overview_Centroid',
        'file_name': system_name,
        'visual_label': 'System Centroid' # Etiqueta diferente para distinguirlos visualmente
    })

print(f"Centroides calculados: {len(centroid_vectors)}")

# Convertimos metadatos de centroides a DataFrame
df_centroids = pd.DataFrame(centroid_metadata)


# --- PASO 3: Unir Todo (Stacking) ---

# A. Unir Vectores (Matrices Numpy)
# Obtenemos matriz de FRs
fr_matrix = np.stack(df_fr['embeddings_mean_se'].values)
# Obtenemos matriz de Centroides
centroid_matrix = np.stack(centroid_vectors)

# Unimos verticalmente: Primero FRs, luego Centroides
final_vectors = np.vstack([fr_matrix, centroid_matrix])

# B. Unir Metadatos (DataFrames)
# Seleccionamos las mismas columnas en ambos
cols_to_save = ['sentence_safe', 'tipo', 'file_name', 'visual_label']
final_metadata = pd.concat([df_fr[cols_to_save], df_centroids[cols_to_save]], ignore_index=True)


# --- PASO 4: Guardar Archivos ---

print(f"Total de puntos a visualizar: {len(final_vectors)}")
print(f"Dimensiones de la matriz: {final_vectors.shape}")

# 1. Guardar Vectores
np.savetxt('vectors_centroids_fr_re.tsv', final_vectors, delimiter='\t')
print("-> vectors_centroids_fr.tsv guardado.")

# 2. Guardar Metadatos
final_metadata.to_csv('metadata_centroids_fr_re.tsv', sep='\t', index=False)
print("-> metadata_centroids_fr.tsv guardado.")