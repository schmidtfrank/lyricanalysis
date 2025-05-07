import pandas as pd
import re
import unicodedata
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob 
import numpy as np

#class to contain data
class LyricSoftware:
    #initialize internal dataset in the constructor
    def __init__(self):
        self.df = pd.read_csv("Songs.csv")
        self.stemmer = PorterStemmer()

    def stem(self):
        for i in self.df.index:
            words = word_tokenize(self.df.iat[i, 2])
            stemmed_words = [self.stemmer.stem(word) for word in words]
            self.df.iat[i, 2] = " ".join(stemmed_words)

    def normalizeText(self,text):
        #handle unicode
        return "".join(c for c in unicodedata.normalize("NFKC", text))
    
    #preprocess
    def preprocess(self):
        replaceDict = {"embedshare urlcopyembedcopy" : "", "\n" : " "}

        punctuation = "1234567890.,?!;:-'\"*()[]"
        pattern = f"[{re.escape(punctuation)}]"
        stop_words = stopwords.words('english')

        for i in self.df.index:
            #lowercase
            self.df.iat[i,2] = self.df.iat[i,2].lower()

            for word in stop_words:
                #remove stop words
                self.df.iat[i,2] = re.sub(r"\b" + re.escape(word) + r"\b", "", self.df.iat[i,2])
        
            #regex remove punctuation chars
            self.df.iat[i,2] = self.normalizeText(self.df.iat[i,2])
            self.df.iat[i,2] = re.sub(pattern, " ",self.df.iat[i,2])
            for key, value in replaceDict.items():
                #now remove bigger phrase and newlines
                self.df.iat[i,2] = self.df.iat[i,2].replace(key,value)
            
            #finally remove all trailing/duplicate white space
            self.df.iat[i,2] = re.sub(r"\s+", " ", self.df.iat[i,2]).strip()
    

    def analyzeSentiment(self):
        """Calculate sentiment polarity for each song lyric and add it as a new column."""
        sentiments = []
        for i in self.df.index:

            lyric = self.df.iat[i, 2]
            # Compute sentiment polarity with TextBlob
            sentiment_score = TextBlob(lyric).sentiment.polarity
            sentiments.append(sentiment_score)
        self.df['Sentiment'] = sentiments
    
    def normalizeSentiment(self):
        self.mu = self.df["Sentiment"].mean()
        self.sigma = self.df["Sentiment"].std()
        self.df["normalizedSentiment"] = (self.df["Sentiment"] - self.mu) / self.sigma
    
    def build_vocabulary(self):  
        counts = self.df.iloc[:,2].str.split().explode().value_counts()
        vocab = np.array(counts.nlargest(200).index)
        self.vocab = vocab
    
    def compute_IDF(self, M, collection):
        self.IDF = np.zeros(self.vocab.size) 
        docCount = {word: 0 for word in self.vocab}

        #check each document and count word appearance
        for doc in collection:
            #make it a set because i was double counting words
            boxOfWords = set(doc.split())
            for word in boxOfWords:
                if word in docCount:
                    docCount[word] += 1
        
        #finally compute each idf
        for i in range(self.vocab.size):
            if docCount[self.vocab[i]] == 0:
                self.IDF[i] = 0
                continue
            self.IDF[i] = np.log(((M+1) / docCount[self.vocab[i]]))
    
    def text2TFIDF(self, text):
        vocab = self.vocab
        tfidfVector = np.zeros(vocab.size)
        
        wordList = text.split()
        for i, word in enumerate(vocab):
            if word in wordList:
                tfidfVector[i] = wordList.count(word)
                tfidfVector[i] = (((3 + 1) * tfidfVector[i]) / (tfidfVector[i] + 3)) * self.IDF[i]
        
        return tfidfVector
    
    def tfidf_score(self, query, doc):
        q = self.text2TFIDF(query)
        d = self.text2TFIDF(doc)
        relevance = np.dot(q,d)
        return relevance
    
    def adapt_vocab_query(self, query):
        queryWords = query.split()
        for word in queryWords:
            if word not in self.vocab:
                self.vocab = np.append(self.vocab,word)

    def execute_search_TF_IDF(self, query):
        self.adapt_vocab_query(query) 
        
        relevances = {}
        
        for index, row in self.df.iterrows():
            doc = row[2]
            relevance = self.tfidf_score(query,doc)
            relevances[index] = relevance 
        return relevances
    
    def queryOnBoth(self, query):
        sentimentScore = TextBlob(query).sentiment.polarity
        normalizedSentiment = (sentimentScore - self.mu) / self.sigma

        relevances = self.execute_search_TF_IDF(query)

        tmpDF = self.df
        tmpDF["sentimentDiff"] = np.abs(tmpDF["normalizedSentiment"] - normalizedSentiment)
        
        largestRel = sorted(relevances, key=relevances.get,reverse=True)[:5]
        return (tmpDF.nsmallest(5, "sentimentDiff"),largestRel)




    



l = LyricSoftware()
l.preprocess()
print("======PRE-STEMMING OUTPUT======\n")
print(l.df.iat[1,2])
print("\n")
l.stem()
print("======POST-STEMMING OUTPUT======\n")
print(l.df.iat[1,2])

l.analyzeSentiment()
print("======DATAFRAME WITH SENTIMENT SCORES======\n")
print(l.df[['Title', 'Sentiment']])


print("Normalized Sentiment Scores")
l.normalizeSentiment()
print(l.df[["Title","Sentiment","normalizedSentiment"]])

print("====SEARCH IN REGARD TO SENTIMENT AND TFIDF====")
l.build_vocabulary()
query = "words not in vocabulary or something"
l.adapt_vocab_query(query)
l.compute_IDF(l.df.shape[0],l.df["Lyrics"])
sentiments, tfidfs = l.queryOnBoth(query)

print(sentiments)
print(tfidfs)