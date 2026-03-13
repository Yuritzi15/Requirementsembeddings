import pandas as pd

csv_path = 'fr_overview_sentences_with_embeddings_mean.csv'
df = pd.read_csv(csv_path)
# drop the unwanted unnamed columns (axis=1 for columns);
# use inplace or reassign so the change persists
if {'Unnamed: 5','Unnamed: 6'}.issubset(df.columns):
    df.drop(['Unnamed: 5','Unnamed: 6'], axis=1, inplace=True)
else:
    # columns may not exist or already removed; ignore silently
    df = df.drop(columns=[c for c in ['Unnamed: 5','Unnamed: 6'] if c in df.columns])

#filter systems with overview and FR 
id_both= (df.groupby('id_file')['tipo']
          .apply(lambda x: {'Overview', 'FR'}.issubset(set(x)))
          )

valid_ids=id_both[id_both].index

filtered_df=df[
    (df['id_file'].isin(valid_ids))&
    (df['tipo'].isin(['Overview','FR']))
    ]

print(filtered_df)
filtered_df.to_csv('filtered.csv')