import math
from collections import Counter
import pandas as pd
class TFIDFComputer:
    def __init__(self, tokenized_documents,vocab):
        self.documents = tokenized_documents
        self.N = len(tokenized_documents)
        self.vocab = vocab
        #self.vocab = sorted(set(token for doc in tokenized_documents for token in doc))
        self.vocab_size = len(self.vocab)
        self.df = self.compute_df()
        self.idf = self.compute_idf()
        self.tfidf_vectors = self.compute_tfidf_vectors()

    def compute_df(self):
        df = {term: 0 for term in self.vocab}
        for doc in self.documents:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1
        return df

    def compute_idf(self):
        return {
            term: math.log(self.N / (1 + df_t))  # se suma 1 para evitar división por cero
            for term, df_t in self.df.items()
        }

    def compute_tf(self, tokens):
        counts = Counter(tokens)
        total = len(tokens)
        return {term: counts.get(term, 0) / total for term in self.vocab}

    def compute_tfidf_vector(self, tokens):
        tf = self.compute_tf(tokens)
        return [tf[term] * self.idf[term] for term in self.vocab]

    def compute_tfidf_vectors(self):
        return [self.compute_tfidf_vector(doc) for doc in self.documents]

    def get_vectors(self):
        return self.tfidf_vectors

    def get_vocab(self):
        return self.vocab

    def get_tfidf_matrix(self):
      tfidf_matrix = self.compute_tfidf_vectors()
      return pd.DataFrame(tfidf_matrix, columns=self.vocab)

