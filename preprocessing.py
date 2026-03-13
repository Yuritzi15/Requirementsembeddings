import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Preprocessing:
    def __init__(self, data):
        self.data = data.astype(str).tolist()  # Convert data to a list of strings
        # perform basic cleaning/normalization immediately
        self.clean_data()
        self.normalize_data()

        # tokenize and then optionally filter/lemmatize
        self.doc_tokenized = self.tokenize_documents()
        # methods below update both self.data and self.doc_tokenized and now
        # return the token lists so we can safely assign if desired
        self.doc_tokenized = self.remove_stopwords(stopwords=stopwords.words('english'))
        self.doc_tokenized = self.lematize_documents(lemmatizer=WordNetLemmatizer())

        # finally build vocabulary from the tokenized documents
        self.vocab = self.build_vocab()
        

    def normalize_data(self):
        self.data = [item.strip().lower() for item in self.data]

    def clean_data(self):
        self.data = [item for item in self.data if item]  # Remove empty strings
        self.data =[re.sub(r'[^\w\s]', '', item) for item in self.data]  # Remove punctuation
        self.data =[re.sub(r'[^a-zA-Z0-9\s]', '', item) for item in self.data]  # Remove special characters
        self.data =[re.sub(r'\s+', ' ', item) for item in self.data]  # Replace multiple spaces with a single space

    def preprocess(self):
        self.clean_data()
        self.normalize_data()
        return self.data
    
    def tokenize_documents(self):
        doc_tokenized =[]
        for doc in self.data:
            tokens = doc.split()
            doc_tokenized.append(tokens)
        return doc_tokenized
    
    def remove_stopwords(self, stopwords):
        """Drop provided stopwords from each document.

        This method updates both ``self.data`` (joined text) and
        ``self.doc_tokenized`` (list-of-list of tokens) and returns the
        tokenized documents for convenience.
        """
        self.data = [' '.join([word for word in doc.split() if word not in stopwords]) for doc in self.data]
        self.doc_tokenized = [doc.split() for doc in self.data]
        return self.doc_tokenized

    def lematize_documents(self, lemmatizer):
        """Lemmatize every word in each document and update internal state.

        Returns the resulting tokenized documents as well.
        """
        self.data = [' '.join([lemmatizer.lemmatize(word) for word in doc.split()]) for doc in self.data]
        self.doc_tokenized = [doc.split() for doc in self.data]
        return self.doc_tokenized

    def build_vocab(self):
        vocab = set()
        for doc in self.doc_tokenized:
            vocab.update(doc)
        return sorted(vocab)
    
    def get_vocab(self):
        return self.vocab
    
        
    def get_tokenized_documents(self):
        return self.doc_tokenized