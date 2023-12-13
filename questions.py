import nltk
import sys
import string
import math
import os

FILE_MATCHES = 5
SENTENCE_MATCHES = 1
PUNCTUATION = string.punctuation
STOPWORDS = nltk.corpus.stopwords.words('english')


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    knowledge = dict()
    for filename in os.listdir(directory):
        file = open(os.path.join(directory,filename),encoding="utf8")
        content = file.read()
        knowledge[filename] = content
    return knowledge
    #raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document)
    words = []
    for i in tokens:
        if i not in PUNCTUATION:
            if i not in STOPWORDS:
                words.append(i.lower())
    return words
    #raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    total_documents = len(documents)
    
    def calc_idf(word):
        word_containing_docs = 0
        for doc in documents:
            if word in documents[doc]:
                word_containing_docs = word_containing_docs + 1
        idf = math.log(total_documents/word_containing_docs)
        return idf

    words_idf = dict()
    for docs in documents:
        for word in documents[docs]:
            if word not in words_idf:
                words_idf[word] = calc_idf(word)
    return words_idf
    #raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf = []
    doclist = []
    for doc in files:
        sum_of_tfidf = 0
        for word in query:
            idf = idfs[word]
            tf = files[doc].count(word)
            sum_of_tfidf = sum_of_tfidf + (tf*idf)
        doclist.append(doc)
        tfidf.append(sum_of_tfidf)
    sortedlist = [x for _,x in sorted(zip(tfidf,doclist))]
    sortedlist = list(reversed(sortedlist))
    return sortedlist[:n]
    #raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf_score = []
    sentences_list = []
    for sentence in sentences:
        score = 0
        for word in query:
            if word in sentences[sentence]:
                score = score + idfs[word]
        sentences_list.append(sentence)
        idf_score.append(score)
    sortedlist = [x for _,x in sorted(zip(idf_score,sentences_list))]
    sortedlist = list(reversed(sortedlist))
    idf_score = sorted(idf_score,reverse = True)
    m = max(idf_score.count(idf_score[0]),n)
    sortedlist = sortedlist[:m]
    idf_score = idf_score[:m]

    qtd = []
    for sentence in sortedlist:
        common_words = set(sentences[sentence]).intersection(query)
        qtd.append(len(common_words)/len(sentences[sentence]))
    sortedlist = [x for _,x in sorted(zip(qtd,sortedlist))]
    sortedlist = list(reversed(sortedlist))

    return sortedlist[:n]
    #raise NotImplementedError


if __name__ == "__main__":
    main()
