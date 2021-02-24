# from fim import apriori, eclat, fpgrowth, fim
from __future__ import print_function

import re
import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from sklearn.externals import joblib

#### ---- download stopwords and punkt ---- ####
nltk.download('stopwords')
nltk.download('punkt')

# define data path here
data_path = './paper_dataset.txt'


# This function filters a 'list of list', so that it only contains 'lists' of length 'size'
def by_size(words, size):
    """Form a list

    Keyword arguments:
    words -- a list of list containing words
    size -- The target length
    """
    return [word for word in words if len(word) == size]


# This function extract the title from data file
def parseTitle():
    """This Function opens the dataset file, and extracts the 'title' field

    Keyword arguments:
    NONE
    """
    # my code here
    with open(data_path) as f:
        lines = f.readlines()
        titles = []
        for line in lines[1:]:
            try:
                temp_titles = line.split('\t')[4]
                titles.append(temp_titles)
            except IndexError:
                continue
        # print(authors)
        return titles


# This function gets extract the abstract from data file
def parseAbstract():
    """This Function opens the dataset file, and extracts the 'Abstract' field

    Keyword arguments:
    NONE
    """
    # my code here
    with open(data_path) as f:
        lines = f.readlines()
        abstracts = []
        for line in lines[1:]:
            try:
                temp_abstracts = line.split('\t')[7]
                # temp_abstracts = line.split('\t')[7][:-3]
                abstracts.append(temp_abstracts)
            except IndexError:
                continue
        return abstracts


def tokenize_and_stem(text):
    # """
    # This Function recieves a string, then tokenize and stem all words in the string
    # Takes in a string of text, then performs the following:
    # 1. Remove all punctuation
    # 2. Remove all stopwords
    # 3. Return the cleaned text as a list of words
    # 4. Remove words
    # Args:
    #     text: the string containing the target text
    #     new_stopwords: the nltk stopwords + additional_stopwords
    # Returns:
    #     filtered_tokens, stemmed_tokens
    # """

    # stemmer = SnowballStemmer("english")
    stemmer_ = WordNetLemmatizer()

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    # 2. Remove all stopwords and more_stopwords
    #stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [word for word in filtered_tokens if word not in stopwords]
    #stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]
    stemmed_tokens = [stemmer_.lemmatize(t) for t in filtered_tokens]
    return stemmed_tokens

def tokenize_only(text, new_stopwords=[]):
    """This Function opens the dataset file, and ONLY tokenize the words

    Keyword arguments:
    text -- contains the target text
    """
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # 2. Remove all stopwords and more_stopwords
    filtered_tokens = [word for word in filtered_tokens if word not in stopwords]
    return filtered_tokens


def text_process(text, more_stopwords=[]):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation and lower-case
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    :param text:
    :param more_stopwords:
    :return:
    """
    stemmer = WordNetLemmatizer()
    # 1. Remove all punctuation and lower-case
    nopunc = text.lower().translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    #nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    # 2. Remove all stopwords and more_stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    if more_stopwords:
        stopwords.extend(more_stopwords)
    nopunc =  [word for word in nopunc.split() if word not in stopwords] #.words('english')]
    return [stemmer.lemmatize(word) for word in nopunc]


def remove_punctuation(text):
    '''
    Remove punctuation from input text string
    :param text:
    :return:
    '''
    s = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))  # None, string.punctuation)
    # print(s)
    return s


if __name__ == "__main__":
    titles = parseTitle()
    abstracts = parseAbstract()

    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")

    totalvocab_stemmed = []
    totalvocab_tokenized = []

    stopwords = nltk.corpus.stopwords.words('english')

    #### ---- Additional Stopwords list to be added ---- ####
    #more_stopwords = ['data', 'set', 'using', 'algorithm', 'approach', 'base', 'method', 'paper', 'present', 'problem',
    #                  'propose', 'results', 'useful']
    # if more_stopwords:
    #     stopwords.extend(more_stopwords)

    # Process text to be ready for clustering. Processing include: removing punctutaion, stopwords, stemming, tokenizing
    for i in abstracts:
        # replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        # i = i.translate(replace_punctuation)
        i = remove_punctuation(i)
        # index list
        words_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(words_stemmed)
        # words list
        words_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(words_tokenized)

    vocab_frame = pd.DataFrame(
        {'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

    # tfidf_vectorizer = TfidfVectorizer(max_df=0.70,
    #                                    min_df=0.30,
    #                                    max_features=200000,
    #                                    stop_words='english',
    #                                    use_idf=True,
    #                                    tokenizer=tokenize_and_stem,
    #                                    ngram_range=(1, 3))
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem,
                                       ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts)  # fit the vectorizer to synopses
    print(abstracts[:5])
    print(tfidf_matrix.shape)
    terms = tfidf_vectorizer.get_feature_names()
    print("# of terms:" + str(len(terms)))
    print(terms)

    dist = 1 - cosine_similarity(tfidf_matrix)

    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()

    papers = {'title': titles, 'abstract': abstracts, 'cluster': clusters}
    frame = pd.DataFrame(papers, index=[clusters], columns=[
        'title', 'abstract', 'cluster'])

    print(frame['cluster'].value_counts())
    print()
    print('======================')
    print("Top terms per cluster:")
    print()

    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')

        for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print()  # add whitespace
        print()  # add whitespace

    # define the linkage_matrix using ward clustering pre-computed distances
    linkage_matrix = ward(dist)

    # Plot the dendogram showing hierirical similarity 
    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=titles, leaf_font_size=1)

    plt.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout

    # uncomment below to save figure
    plt.savefig('ward_clusters_new.png', dpi=800)  # save figure as ward_clusters
