import os

import pandas as pd
import numpy as np
from preprocessing import Preprocessing
from tfidf import TFIDFComputer
import matplotlib.pyplot as plt
from CountWords import CountWords
from llms import LLMFactory
from llms import GeminiFactory
from llms import OllamaFactory

def plot_pareto_curve(probabilities, system_name):
    #order max to min 
    values=probabilities.sort_values(ascending=False)
    
    if len(values) == 0:
        print(f"No probabilities for: {system_name}")
        return
    
    cumulative_sum = values.cumsum()

    threshold =0.8
    cutoff_index=(cumulative_sum>=threshold).idxmax()
    cutoff_position=cumulative_sum.index.get_loc(cutoff_index) + 1

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, 
              len(cumulative_sum) + 1), 
              cumulative_sum, 
              marker='o', 
              color="#330CDD", 
              linewidth=2
             )
    plt.axhline(y=0.8, color='r', linestyle='--')  # Add a horizontal line at 80%
    plt.xlabel("Number of Words")
    plt.ylabel("Cumulative Probabilties")
    plt.title("Pareto Curve of word probabilites for system: " + system_name)
    dir="pareto_curves"
    os.makedirs(dir, exist_ok=True)
    safe_system_name = "".join([c if c.isalnum() or c in (' ', '_', '-') else '_' for c in system_name])
    safe_system_name = safe_system_name[:100]
    name=f"pareto_curve_{safe_system_name}.png"
    save_path = os.path.join(dir, name)
    plt.savefig(save_path)
    plt.close()
    
    relevant_words=values.iloc[:cutoff_position]

    return relevant_words
    # plt.show()


def consult_llm(factory:LLMFactory, prompt:str):
    text_generator=factory.create_text_generator()

    print(f"prompt: {prompt}")

    result=text_generator.query(prompt)
    print(f'Response: {result}')
    print("-"*40)



def main():
    # Load data
    df = pd.read_csv('filtered.csv')
    df=df[['sentence','id_file','file_name','tipo']]
    df=df[df['tipo']=='Overview']

    API_KEY="AIzaSyAm6kU34gkjUZqDgKX8rfQCWVUIG33_waI"
    gemini_factory=GeminiFactory(api_key=API_KEY)
   

    # Preprocess overview texts
    preprocessor = Preprocessing(df['sentence'])
    preprocessed_overviews = preprocessor.get_tokenized_documents()
    print("Preprocessing completed. Sample preprocessed overview:", len(preprocessed_overviews))
    vocab = preprocessor.get_vocab()
    print("Vocabulary size:", len(vocab))

    cw=CountWords()
    df['tokens']=preprocessed_overviews
    #print(df)
    top_words=[]
    systems=df['file_name'].unique()
    for sistem in systems:
        df_temp=df[df['file_name']==sistem]
        # print(df_temp['tokens'])
        probabilities=cw.wordsinsoverview(df_temp['tokens'])
        relevant_words=plot_pareto_curve(probabilities,sistem).index.tolist()
        top_words.append(relevant_words)
        #print(top_words)
    
    df_spech_tree = pd.DataFrame()
    df_spech_tree['file_name']=systems
    df_spech_tree['level1']=top_words
    print(df_spech_tree)

    prompt="Dime 10 palabras que estén altamente relacionadas con"
    consult_llm(gemini_factory,prompt=prompt)


    



    #print(len(df['file_name'].unique()))
    # # tokens = preprocessor.get_tokens()
    # # print("Sample tokens for the first tokens:", tokens[:10])  # Debug: Print the first 10 tokens to check the tokenization process

    # #get the matrix of TF-IDF vectors
    # tfidf_computer = TFIDFComputer(preprocessed_overviews, vocab)
    # tfidf_matrix = tfidf_computer.get_tfidf_matrix()

    # print(tfidf_matrix.head())  # Debug: Print the first few rows of the TF-IDF matrix

    # save_path = 'tfidf_matrix.csv'
    # tfidf_matrix.to_csv(save_path, index=False)

    # # lambda function to get the highest terms by document
    # top_terms = tfidf_matrix.apply(
    #     lambda row: highest_tfidf_terms_by_row(row, threshold=0.8), axis=1
    # )
    # print("Top TF-IDF terms for each document:")
    # print(top_terms.head())  # Debug: Print the top terms for the first few documents
    # df['top_tfidf_terms'] = top_terms
    # # df.to_csv('overview_sentences_with_top_terms.csv', index=False)
    # #80 is enouh?
    # #get the pareto curve to each document 
    # for index, row in tfidf_matrix.iterrows():
    #     system_name = df.iloc[index]['file_name']  # Assuming 'file_name' is the column with system names
    #     plot_pareto_curve(row, system_name)

if __name__ == "__main__":    
    main()