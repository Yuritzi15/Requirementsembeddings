import re

class Preprocessing:
    def __init__(self, data):
        self.data = data.astype(str).tolist()  # Convert data to a list of strings
        

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