import pandas as pd
from preprocessing import Preprocessing
from bertembedder import BertEmbedder
from tqdm import tqdm


df = pd.read_csv('fr_overview_sentences.csv')
prep = Preprocessing(df['sentence'])
# New column to store the processed sentences
df['processed_sentences'] = prep.preprocess()
print(df.head())

embedder = BertEmbedder()
embeddings_cls = []
embeddings_mean = []
for sentence in tqdm(df['processed_sentences'], desc='Embedding', total=len(df)):
    emb_mean=embedder.embed(sentence)
    #emb_cls, emb_mean = embedder.embed(sentence)
    #embeddings_cls.append(emb_cls.squeeze().numpy())
    embeddings_mean.append(emb_mean.squeeze().numpy())
#df['embeddings_cls'] = embeddings_cls
df['embeddings_mean'] = embeddings_mean
print(df.head())
df.to_csv('fr_overview_sentences_with_embeddings.csv', index=False)