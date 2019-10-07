### Imports
import pandas as pd
import re
import string
import gensim
import datetime
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')





### Constants
path_to_scrapped_corpus = './data/corpus.csv'

pos_lem = {
    'NN': 'n',
    'NNS': 'n',
    'RB': 'r',
    'JJ': 'a',
    'VB': 'v',
    'VBD': 'v',
    'VBG': 'v',
    'VBN': 'v',
    'VBP': 'v',
    'VBZ': 'v',
}



###Main function
def main_preprocessing(run_preprocessing=False):
    if run_preprocessing:
        corpus = pd.read_csv("./data/corpus.csv")

        print("-- Preprocessing :")
        prev_time = measure_time_step(0, True) # Time
        print("Nb of documents", corpus.shape[0])
        corpus = corpus[corpus['text'].apply(cleaning_filter)].reset_index(drop=True)
        print("Nb of documents", corpus.shape[0])
        print(" - Manage structured abstract :")
        corpus['text'] = corpus['text'].apply(str).apply(managed_structured)
        corpus = corpus.drop_duplicates(subset='text')
        print(" - Tokenization, Lemming :")
        corpus['Title'] = corpus['Title'].apply(str).apply(text_preprocessing)
        corpus['text'] = corpus['text'].apply(str).apply(text_preprocessing)
        prev_time = measure_time_step(prev_time) # Time

        corpus = corpus.loc[:, ['article_ID', 'Title', 'Keywords', 'text', 'category']]

        print("-- Remove word that appear to much or to few :")
        print(np.unique(np.concatenate(corpus['text'].values)).shape[0])
        corpus['text'] = filter_out_to_few(corpus['text'])
        print(np.unique(np.concatenate(corpus['text'].values)).shape[0])
        prev_time = measure_time_step(prev_time) # Time

        prev_time = measure_time_step(0, True)  # Time
        print("-- Train model :")
        model = train_w2v_model(corpus).wv
        prev_time = measure_time_step(prev_time)  # Time

        corpus_prep = corpus.copy()

        prev_time = measure_time_step(0, True)  # Time

        print("-- Vectorize :")
        corpus_prep = vectorize_corpus(corpus_prep, methods=[
            'w2v', "tfidf", 'tfidf_w2v_concat', 'w2v_tfidf', 'tfidf_w2v_tfidf_concat'
        ], model=model)
        prev_time = measure_time_step(prev_time)  # Time

        corpus_prep = corpus_prep[corpus_prep['w2v_tfidf'].apply(lambda x: not(None in x))].reset_index()

        corpus_prep.to_json("./data/corpus_prep.zip")
    else:
        corpus_prep = pd.read_json("./data/corpus_prep.zip")
    return corpus_prep






### Core functions
def measure_time_step(prev_time, init=False):
    """
    Convinient way to measure time in the execution flow
    :param prev_time: end time of the previous execution
    :return: current time
    """
    current = datetime.datetime.now()
    if not (init):
        print(current - prev_time)
    return current


def cleaning_filter(text):
    """
    Filter to get rid of retracted and badly formatted article
    :param text: abstract
    :return: Pandas Serie True (keep) are False (delete)
    """
    try:# A REFAIRE
        if "This article has been retracted" in text:
            text ="retracted"
            return False
        if "Cette article" in text:
            text ="retracted"
            return False
        if len(text) < 20:
            return False
    except:
        return False
    return True

def managed_structured(text):
    """
    Transform 'StringElement()' text to normal text
    :param text: text to parse
    :return: parsed text
    """
    text_search = re.search("StringElement\(\\\'(.*?)\\\', attributes=", text)
    if text_search:
        return " ".join(text_search.groups())
    else:
        return text

def text_preprocessing(text, word_reduction='lemmatization', pos_lem=pos_lem):
    """
    Preprocess a text (stop word punctuation lemmatization and tokenization ...)
    :param text: text to preprocess
    :param word_reduction: lemmatization or stemming
    :param pos_lem: dictionnary rule of pos lemmatization
    :return: tokenized preprocess text
    """
    # To lower case
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            '', '',
            string.punctuation + '–±·'
        )
    )

    # Remove whitespace
    text = text.strip()

    # Remove stop word and words less than 2 letter long
    stopword_en = stopwords.words("english")
    tokens = word_tokenize(text)
    tokens = list(filter(lambda x: not (x in stopword_en) and len(x) > 2, tokens))

    # Keep only the POS we want to keep
    tokens_pos = nltk.pos_tag(tokens)
    if len(pos_lem.keys()) > 0:
        tokens_pos = list(filter(lambda x: pos_lem.get(x[1], False), tokens_pos))

    if word_reduction == 'stemming':
        stemmer = PorterStemmer()
        tokens = list(map(lambda x: stemmer.stem(x[0]), tokens_pos))
    if word_reduction == 'lemmatization':
        lemmatizer = WordNetLemmatizer()
        tokens = list(map(lambda x: lemmatizer.lemmatize(x[0], pos=pos_lem[x[1]]), tokens_pos))

    return tokens


def filter_out_to_few(corpus):
    """
    Remove word that appear too much (not pertinent) and too few (not useful)
    :param corpus: dataframe of the corpus
    :return: corpus dataframe without the words we want to remove
    """
    # Remove words that appear less than 5 time in the whole corpus
    words = list(np.concatenate(corpus.values))
    word_count = dict(Counter(words))
    corpus = corpus.apply(lambda x: list(filter(lambda y: word_count[y] > 5, x)))

    # Remove words that appear in more than 50% of the documents
    words = list(np.concatenate(
        corpus.apply(lambda x: list(set(x))).values
    ))
    word_appearance_count = dict(Counter(words))
    nb_doc = corpus.shape[0]
    corpus = corpus.apply(
        lambda x: list(filter(lambda y: word_appearance_count[y] < nb_doc / 2, x))
    )

    return corpus


def w2v_get_vector(word, model=None):
    """
    Get the vector from the w2v model (returning None if the word is absent)
    :param word: the word we want to transform to a vector
    :param model: the model of w2v
    :return: vector
    """
    try:
        return model.get_vector(word)
    except Exception as e:
        return None


def vectorisation_w2v(tokens, agg='mean', model=None, word_coefficients=None):
    """
    Vectorize the tokenized text
    :param tokens: tokenized text
    :param agg: type of aggregation (tfidf is a mean weighted by tfidf)
    :param model: the w2v model
    :param word_coefficients: coefficient from the tfidf if agg="tfidf"
    :return: vector of the text
    """
    # Each word to w2v
    token_words = tokens
    tokens = list(map(
        lambda x: w2v_get_vector(x, model), tokens
    ))
    tokens = list(filter(lambda x: str(x) != 'None', tokens))
    tokens = np.array(tokens)
    tokens_w_w2v = list(zip(token_words, tokens))

    # Aggregation
    if agg == 'mean':
        tokens = np.mean(tokens, axis=0)
    if agg == 'sum':
        tokens = np.sum(tokens, axis=0)
    if agg == 'tfidf':
        # print(word_coefficients)
        tokens = np.sum(list(map(
            lambda x: x[1] * float(word_coefficients.get(x[0]) or 0),
            tokens_w_w2v
        )), axis=0)
        tokens = tokens / sum(word_coefficients.values())
    return tokens


def vectorize_corpus(corpus, methods=["w2v", "tfidf"], model=None):
    """
    Vectorize the corpus
    :param corpus: our corpus dataframe
    :param methods: list of the vectorisation methods we want to have
    :param model: w2v model we want to use
    :return: previous dataframe with one columns more per vectorisation
    """
    corpus = corpus.reset_index(drop=True)
    if "tfidf" in methods:
        # Do the TFIDF
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,  # already tokenized
            preprocessor=lambda x: x,  # already tokenized
            max_features=500,
            token_pattern=None
        )
        fitted_tfidf = vectorizer.fit_transform(corpus['text'])
        corpus['tfidf'] = pd.Series(fitted_tfidf.todense().tolist())
        corpus['tfidf_features'] = ";".join(vectorizer.get_feature_names())
        corpus['tfidf_features'] = corpus['tfidf_features'].apply(lambda x: x.split(';'))
    # Word2Vec mean vectorization
    if "w2v" in methods:
        corpus['w2v'] = corpus['text'].apply(
            lambda x: vectorisation_w2v(x, agg='mean', model=model)
        )
    # Concatenation with word2vec and tfidf
    if ("w2v" in methods) and ("tfidf" in methods) and ("tfidf_w2v_concat" in methods):
        corpus['tfidf_w2v_concat'] = corpus.apply(
            lambda x: np.concatenate([x['w2v'], x['tfidf']]), axis=1
        )
    # Word2Vec weighted mean using tfidf vectorization
    if ("w2v_tfidf" in methods) and ("tfidf" in methods):
        corpus['w2v_tfidf'] = corpus.apply(
            lambda x: vectorisation_w2v(
                x['text'],
                agg='tfidf',
                model=model,
                word_coefficients=dict(zip(x['tfidf_features'], x['tfidf']))
            ),
            axis=1
        )
    # Concatenation with word2vec meaned with tfidf and tfidf
    if ("w2v_tfidf" in methods) and ("tfidf" in methods) and ("tfidf_w2v_tfidf_concat" in methods):
        corpus['tfidf_w2v_tfidf_concat'] = corpus.apply(
            lambda x: np.concatenate([x['w2v_tfidf'], x['tfidf']]), axis=1
        )
    return corpus


def train_w2v_model(corpus):
    """
    Train a w2v using our corpus
    :param corpus: dataframe of our corpus
    :return: gensim w2v model
    """
    model = gensim.models.Word2Vec(corpus['text'].values, size=300, window=5, min_count=5, workers=4)
    model.train(corpus['text'].values, total_examples=corpus['text'].shape[0], epochs=500)
    return model


