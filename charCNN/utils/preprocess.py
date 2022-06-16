import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def preprocess_text(
    train_path:str,
    test_path:str
) -> (list[str], list[str], list[list[int]], list[list[int]]):

    tk = make_tokenizer()
    maxlen = np.nan
    results = [None] * 5

    for idx, path in enumerate([train_path, test_path]):
        df = read_df(path)
        
        if path is train_path:
            # maxlen = df['text'].str.len().max()
            maxlen=1014

        # while(np.isnan(maxlen)):
        #     continue

        results[idx*2] = convert_text(df, tk, maxlen)
        results[(idx*2)+1] = get_class_list(df)

    results[4] = tk
    
    return results


def read_df(path:str) -> pd.DataFrame:

    # Read in data
    df = pd.read_csv(path, header=None, names=['class', 'title', 'description'])

    # concatenate column 1 and 2 as one text
    df['text'] = df.title + df.description

    return df.drop(['title', 'description'], axis=1, inplace=False)


def make_tokenizer():

    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')

    # Although we generated a vocabulary already, we already have an existing character list:
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1

    tk.word_index = char_dict   # assign Tokenizer's word index to our custom index
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1   # append 'UNK' to be the next sequential value of the char dict

    return tk


def convert_text(
    df:pd.DataFrame,
    tk,
    maxlen:int
) -> list[list[str]]:

    texts = [ s.lower() for s in df.text.values ] # preproc texts to all lowercase to match vocab
    sequences = tk.texts_to_sequences(texts)

    data = pad_sequences(
        sequences,          # sequences to be padded
        maxlen=maxlen,      # get max length of all sequences
        padding='post'      # pad sequences on the right end
    )

    return data


def get_class_list(df:pd.DataFrame) -> list[list[int]]:

    class_list = [ x-1 for x in df['class'].values ]

    return to_categorical(class_list)