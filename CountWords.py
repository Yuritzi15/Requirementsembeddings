import pandas as pd 


class CountWords:
    
    def __init__(self):
        pass


    def wordsinsoverview(self, text:pd.Series):
        vocabulary=text.explode()
        word_counts=vocabulary.value_counts()
        probability=word_counts/word_counts.sum()
        return probability

