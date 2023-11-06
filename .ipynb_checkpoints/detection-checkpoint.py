import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer  # love = [0.0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# converting in base word loved,loving into love


news_df = pd.read_csv('train.csv')
print(news_df.head())

# preprocessing removing noisy data
print(news_df.shape)
print(news_df.isna().sum())  # to see how many null values we have in dataset
# replacing null value is not possible so we will replace it with ' '
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author']+" "+news_df['title']
# print(news_df)

# removing noisy data in a dataset is known as stemming removing . , : ' capital letters etc
ps = PorterStemmer()


def stemming(content):
    # ^ stands for negotiation
    stemmed_content = re.sub('[^a-zA-Z]', " ", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(
        word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)


news_df['content'] = news_df['content'].apply(stemming)

