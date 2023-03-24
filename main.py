import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('SnowballStemmer')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import string

#Making a dataframe

df = pd.read_csv('labeled.csv', sep=',')
df['toxic'] = df['toxic'].apply(int)
train_df, test_df = train_test_split(df, test_size=500)
#Text Preprocessing

snowball = SnowballStemmer(language = 'russian')
rswords = stopwords.words('russian')

def tokenize(sentence: str, remove_swords: bool=True):
    tokens_begin = word_tokenize (sentence, language = 'russian')
    tokens_wout_punct = [i for i in tokens_begin if i not in string.punctuation]
    tokens_wout_punct_and_rswords = [i for i in tokens_wout_punct if i not in rswords]
    tokens = [snowball.stem(i) for i in tokens_wout_punct_and_rswords]
    return tokens

vectorizer = TfidfVectorizer (tokenizer = lambda x: tokenize(x))
features = vectorizer.fit_transform(train_df['comment'])
#Model which is learning on a vectors
# model = LogisticRegression(random_state=0)
# model.fit(features, train_df['toxic'])
#Model which is learning on a sentences
model_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer (tokenizer = lambda x: tokenize(x))),
    ('model', LogisticRegression(random_state=0))
])
model_pipeline.fit(train_df['comment'], train_df['toxic'])

#Metrics

print(precision_score(y_true=test_df['toxic'], y_pred=model_pipeline.predict(test_df['comment'])))
print(recall_score(y_true=test_df['toxic'], y_pred=model_pipeline.predict(test_df['comment'])))

flag=True
while flag:
    print('Your comment')
    com = str(input())
    if model_pipeline.predict([com])==0:
        print("It's ok")
    else:
        print("It's not ok")
    print('Continue? y/n')
    ans = input()
    if ans == 'n':
        flag = False