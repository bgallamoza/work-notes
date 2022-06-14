import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def preprocess_text(
    train:str,
    test:str
) -> (list[str], list[str], list[list[int]], list[list[int]]):

    return


def read_df(path:str) -> pd.DataFrame:

    # Read in data
    df = pd.read_csv(path, header=None, names=['class', 'title', 'description'])

    # concatenate column 1 and 2 as one text
    df['text'] = df.title + df.description

    return df.drop(['title', 'description'], axis=1, inplace=False)


def make_tokenizer():

    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')

    # Although we generated a vocabulary already, we already have an existing character list:
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    tk.word_index = char_dict   # assign Tokenizer's word index to our custom index
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1   # append 'UNK' to be the next sequential value of the char dict

    return tk


def convert_text(
    df:pd.DataFrame,
    tk:keras.preprocessing.text.Tokenizer
) -> list[list[str]]:

    texts = [ s.lower() for s in df.text.values ] # preproc texts to all lowercase to match vocab
    sequences = tk.texts_to_sequences(texts)