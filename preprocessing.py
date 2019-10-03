### Imports
import pandas as pd
import re
import string
import gensim
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


### Core functions
def cleaning_filter(text):
    '''

    :param text:
    :return:
    '''
  try:# A REFAIRE
    if "StringElement" in text: # A REVOIR SI EUX L'ON GARDE
      text = "badformat"
      return False
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

def text_preprocessing(text, word_reduction='lemmatization', pos_lem=pos_lem):
    '''

    :param text:
    :param word_reduction:
    :param pos_lem:
    :return:
    '''
  # To lower case
  text = text.lower()
  
  # Remove numbers
  text = re.sub(r'\d+', '', text)
  
  # Remove punctuation
  text = text.translate(
      str.maketrans(
          '', '',
          string.punctuation +  '–±·'
      )
  )
  
  # Remove whitespace
  text = text.strip()
  
  # Remove stop word and words less than 2 letter long
  stopword_en =  stopwords.words("english")
  tokens = word_tokenize(text)
  tokens = list(filter(lambda x: not(x in stopword_en) and len(x)>2, tokens))
  
  # Keep only the POS we want to keep
  tokens_pos = nltk.pos_tag(tokens)
  if len(pos_lem.keys()) > 0:
    tokens_pos = list(filter(lambda x: pos_lem.get(x[1], False), tokens_pos))
  
  if word_reduction == 'stemming':
    stemmer = PorterStemmer()
    tokens = list(map(lambda x: stemmer.stem(x[0]), tokens_pos))
  if word_reduction == 'lemmatization':
    lemmatizer=WordNetLemmatizer()
    tokens = list(map(lambda x: lemmatizer.lemmatize(x[0], pos=pos_lem[x[1]]), tokens_pos))
  
  return tokens


def filter_out_to_few(corpus):
    '''

    :param corpus:
    :return:
    '''
  #Remove words that appear less than 5 time in the whole corpus
  words = list(np.concatenate(corpus.values))
  word_count = dict(Counter(words))
  corpus = corpus.apply(lambda x: list(filter(lambda y: word_count[y]>5, x))) 
  
  #Remove words that appear in more than 50% of the documents
  words = list(np.concatenate(
      corpus.apply(lambda x: list(set(x))).values
  ))
  word_appearance_count = dict(Counter(words))
  nb_doc = corpus.shape[0]
  corpus = corpus.apply(
      lambda x: list(filter(lambda y: word_appearance_count[y]<nb_doc/2, x))
  )
  
  return corpus

def w2v_get_vector(word):
    '''

    :param word:
    :return:
    '''
  try:
    return model.get_vector(word)
  except Exception as e:
    return None

def vectorisation_w2v(tokens, agg='mean', model=None, word_coefficients=None):
    '''

    :param tokens:
    :param agg:
    :param model:
    :param word_coefficients:
    :return:
    '''
  tokens = list(map(w2v_get_vector, tokens))
  #tokens = list(map(lambda x: testtt[x], tokens))
  tokens = list(filter(lambda x: str(x) != 'None', tokens))
  tokens = np.array(tokens)
  if agg == 'mean':
    tokens = np.mean(tokens, axis=0)
  if agg == 'sum':
    tokens = np.sum(tokens, axis=0)
  if agg == 'tfidf':
    tokens = np.sum(np.array(map(w2v_get_vector(tokens)*word_coefficients[x] , tokens)), axis=0)
  return tokens


def vectorize_corpus(corpus, methods=["w2v", "tfidf"], model = model2):
    '''

    :param corpus:
    :param methods:
    :param model:
    :return:
    '''
  corpus = corpus.copy()
  if "tfidf" in methods:
    vectorizer = TfidfVectorizer(
      tokenizer=lambda x: x, # already tokenized
      preprocessor=lambda x: x, # already tokenized
      max_features=500,
      token_pattern=None
    )
    
    fitted_tfidf = vectorizer.fit_transform(corpus['text'])
    corpus['tfidf'] =  pd.Series(fitted_tfidf.todense().tolist())
    corpus['tfidf_features'] = ";".join(vectorizer.get_feature_names())
    corpus['tfidf_features'] = corpus['tfidf_features'].apply(lambda x: x.split(';'))
  if "w2v" in methods:
    corpus['w2v'] =  corpus['text'].apply(
        lambda x: vectorisation_w2v(x, agg='mean', model=model)
    )
  if "w2v_tfidf" in methods:
    corpus['w2v'] =  corpus['text'].apply(
        lambda x: vectorisation_w2v_tfidf(x, agg='tfidf', model=model)
    )
  return corpus





  ### Execution flow (to be moved to main)
model2 = gensim.models.Word2Vec(corpus['text'].values, size=300, window=5, min_count=5, workers=4)

model2.train(corpus['text'].values, total_examples=corpus['text'].shape[0], epochs=500)
print("heart")
print(model2.wv.most_similar("heart", topn=10))
print("therapeutic")
print(model2.wv.most_similar("therapeutic", topn=10))


  %%time
corpus = pd.read_csv('/content/drive/My Drive/data_colab/corpus.csv')

corpus.sample(n=100)


print("Nb of documents", corpus.shape[0])
corpus = corpus[corpus['text'].apply(cleaning_filter)].reset_index(drop=True)
print("Nb of documents", corpus.shape[0])
corpus['Title'] = corpus['Title'].apply(str).apply(text_preprocessing)
corpus['text'] = corpus['text'].apply(str).apply(text_preprocessing)

# Drop duplicates

corpus = corpus.loc[:, ['article_ID', 'Title', 'Keywords', 'text', 'category']]

print(np.unique(np.concatenate(corpus['text'].values)).shape[0])
corpus['text'] = filter_out_to_few(corpus['text'])
print(np.unique(np.concatenate(corpus['text'].values)).shape[0])

corpus.head()

vectorize_corpus(corpus.sample(n=5), methods=["w2v", "tfidf"])
