import pandas as pd
import re
import unicodedata
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob  # For sentiment analysis



#TODO
#word stemming
#use wordblob? for sentiment analysis on lyrics


#future?
#integrate genius api for song searching based on our "hub"
#things


#class to contain data
class LyricSoftware:
    #initialize internal dataset in the constructor
    def __init__(self):
        self.df = pd.read_csv("Songs.csv")
        self.stemmer = PorterStemmer()
        pass

    #TODO word stemming
    #probably use nltk.stem?
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

