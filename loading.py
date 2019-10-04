import pandas as pd
import numpy as np
import gensim
import nltk
import warnings
import re

from Bio import Entrez
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from gensim.utils import simple_preprocess
from gensim import models
from gensim.parsing.preprocessing import STOPWORDS

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

warnings.filterwarnings("ignore")


# ## 2 - Extracting abstracts

# In order to extract the articles, we used Biopython, the best-known Python library to process biological data.
# 
# If you want to skip the extraction section, you can start running the code from the cell 'Get abstracts'. The file <i> CS2_Article_Clustering.xlsx </i> is provided. 

# In[2]:


def search(query):
    """ search articles with the key word 'query'.
    The database is specified in the parameter 'db'.
    The number of retrived articles is specified in the parameter 'retmax'. 
    The reason for declaring YOUR_EMAIL address is to allow the NCBI to
    contact you before blocking your IP, in case you’re violating the guidelines.
    """
    Entrez.email = 'YOUR_EMAIL'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='1000',
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results


# In[3]:


def fetch_details(id_list):
    """ Fetch details of a list of articles IDs.
    The reason for declaring YOUR_EMAIL address is to allow the NCBI to
    contact you before blocking your IP, in case you’re violating the guidelines.
    """
    ids = ','.join(id_list)
    Entrez.email = 'YOUR_EMAIL'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results


# In[4]:


def collect_data(categories):
    """ Get abstracts for each category in 'categories'. 
    
    return: abstracts
    ----------
    article_ID : ID of the article
    text : abstract of the article if it exists
    category : category of the article
    structured : indicates whether the article is structured
    Keywords : keywords of the article if it exists
    Title : title of the article
    """
    abstracts = pd.DataFrame(columns=['article_ID', 'Title',
                                      'Keywords', 'text', 'category',
                                      'structured'])
    i = 0
    for cat in categories:
        i +=1
        print("Collecting category", str(i), "/", str(len(categories)),":", cat, "...")
        results = search(cat)  # get the articles for the category 'cat'
        id_list = results['IdList']  # select the IDs
        if (len(id_list) > 0):
            papers = fetch_details(id_list)  # get details of articles
            pubmed_articles = papers['PubmedArticle'] 
            for pubmed_article in pubmed_articles:
                s = 1  # structured article
                MedlineCitation = pubmed_article['MedlineCitation']
                pmid = int(str(MedlineCitation['PMID']))
                article = MedlineCitation['Article']
                keywords = MedlineCitation["KeywordList"]
                title = MedlineCitation['Article']['ArticleTitle']
                if(len(keywords) > 0):
                    keywords = list(keywords[0])
                if('Abstract' in article):
                    abstract = article['Abstract']['AbstractText']
                    if(len(abstract) == 1):
                        abstract = abstract[0]
                        s = 0
                else:
                    abstract = ''
                abstracts = abstracts.append({'article_ID': pmid, 'text': abstract,
                                              'category': cat, 'structured': s,
                                              'Keywords': keywords, 'Title': title},
                                             ignore_index=True)  # store the abstract
    return abstracts




# ## 3 - Cleaning & Pre-processing data

# ### 3.1 - Cleaning the data

# In[10]:


def concat_structured_abstracts(abstract):
    """Concat the strings of a list.
    Actually, there are some structured abstracts.
    Two examples of structured and non-structured abstract: 
    structured abstract: https://www.ncbi.nlm.nih.gov/pubmed/30348261
    non-structured abstract: https://www.ncbi.nlm.nih.gov/pubmed/19699449 """
    concat = ""
    for i in range(0, len(abstract)):
        concat = concat + str(abstract[i]) + " "
    return concat


# In[11]:


def concat_keywords(keywords):
    """ concatenate keywords. 
    The type of each keyword is initially 'Bio.Entrez.Parser.StringElement' ans not a string. """
    concat = ""
    for i in range(0, len(keywords)):
        concat = concat + str(keywords[i]) + ", "
    return concat




def count_words(abstract):
    """ Count words of abstract. """
    words = [w for w in word_tokenize(abstract) if nonPunct.match(w)]
    return len(words)


# In[14]:


def count_characters(abstract):
    """ Count characters of abstract. """
    characters = len(abstract)
    return characters


def main_loading():
    # here are defined categories for which we want articles 
    categories = ['cancérologie', 'cardiologie', 'gastro',
                  'diabétologie', 'nutrition', 'infectiologie',
                  'gyneco-repro-urologie', 'pneumologie', 'dermatologie',
                  'industrie de santé', 'ophtalmologie']

    # call the function collect_data to get the abstracts
    abstracts = collect_data(categories)

    # standardize abstracts
    abstracts['text_standardized'] = abstracts.apply(lambda row: concat_structured_abstracts(row['text']) 
                                               if row['structured'] == 1 else row['text'], axis=1)

    # concatenate keywords per abstract. 
    abstracts['list_keywords'] = abstracts.apply(lambda row: concat_keywords(row['Keywords']) 
                                               if len(row['Keywords']) > 0 else row['Keywords'], axis=1)


    # In[13]:


    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    # count number sentences per abstract
    abstracts['nb_sentences'] = abstracts.apply(lambda row: len(sent_tokenize(row['text_standardized'])) , axis=1)

    # count number words per abstract
    abstracts['nb_words'] = abstracts.apply(lambda row: count_words(row['text_standardized']) , axis=1)

    # count number characters per abstract
    abstracts['nb_characters'] = abstracts.apply(lambda row: count_characters(row['text_standardized']) , axis=1)


    # In[16]:


    # construction of df_abstracts
    df_abstracts = abstracts[['article_ID', 'Title', 'text_standardized', 'list_keywords', 'category', 'nb_sentences', 'nb_words', 'nb_characters']]
    df_abstracts.rename(columns={'text_standardized':'text', 'list_keywords':'Keywords'}, inplace=True)


    # In the cell below, we save the abstracts in an Excel file.

    # In[17]:


    # save abstracts
    df_abstracts.to_excel('abstracts.xlsx', index=None)
    return df_abstracts