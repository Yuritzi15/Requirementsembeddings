import preprocessing
import pandas as pd


df=pd.read_csv('fr_overview_sentences.csv')
df['processed_sentences']=df['sentences'].apply(preprocessing.preprocess_text)
print(df.head())