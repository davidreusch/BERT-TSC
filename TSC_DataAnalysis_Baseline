#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib. pyplot as plt  
import seaborn as sns

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore") 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from tabulate import tabulate

from collections import  Counter

from sklearn.feature_extraction.text import CountVectorizer

import re
import random
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer


# In[2]:


df_tr = pd.read_csv('Data/train.csv', delimiter=',')
df_tst = pd.read_csv('Data/test.csv', delimiter=',')
df_tst_labels = pd.read_csv('Data/test_labels.csv', delimiter=',')


# In[3]:


# Creates test samples needed for loss function -> Removes samples with values [-1, -1,...]
df = pd.concat([df_tst, df_tst_labels], axis=1)

df = df[df.toxic != -1]
df = df.reset_index(drop=True) 

test_labels = df.iloc[:, 3:9]
test = df.iloc[:, :2]

df.head()


# In[4]:


#Compute size of each class
df_labels=df_tr.iloc[:, 2:8]
df_labels['non-toxic'] = pd.Series([int(sum(df_labels.iloc[i]) ==0) for i in range(len(df_tr))])
counts = df_labels.apply(pd.value_counts)
counts_list = counts.values.tolist()[1]

counts


# In[5]:


len(df_tr)


# ## Preprocessing

# In[6]:


def preprocess_data(dataframe : pd.DataFrame, colname : str, keep_all = False):
    """
    Applies basic preprocessing (such as dropping invalid samples) to the provided dataset
    
    Parameters:
        dataframe (pd.DataFrame) : The input data consisting of text and label
        keep_all (bool) : If true, no invalid samples will be removed (e.g. True when preprocessing inputs that should be predicted)
    Returns:
        pd.DataFrame : The preprocessed dataset
    """

    df1=dataframe.copy()
    # Ensure type of column
    df1 = df1.astype({colname: str})
    #drop NA
    if not keep_all: 
        df1=df1.dropna()
    # lowcasing
    df1 = df1.apply(lambda x: x.astype(str).str.lower())
    # Remove newlines from messy strings
    df1  = df1.replace(r'\r+|\n+|\t+',' ', regex=True)
    
    #Remove special characters
    df1[colname] = df1[colname].apply(lambda x: re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", str(x)))
    #Punctuation
    df1[colname] = df1[colname].str.replace('[^\w\s]','')
    #Remove digits
    df1[colname] = df1[colname].str.replace('\d+', '')
    #Remove more than one white space
    df1[colname] = df1[colname].apply(lambda x: re.sub(' +', ' ', str(x)))

    # Remove Stopwords
    stop = stopwords.words('english')
    df1[colname] = df1[colname].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    stemmer = SnowballStemmer("english")
    df1[colname] = df1[colname].apply(lambda x: ''.join([stemmer.stem(y) for y in x]))
    lmtzr = WordNetLemmatizer()
    df1[colname] = df1[colname].apply(
                        lambda lst:''.join([lmtzr.lemmatize(word) for word in lst]))

    # Remove character if chr < 1
    df1[colname] = df1[colname].apply(lambda x: re.sub(r'\b\w{1,1}\b', '', x))

    if not keep_all:
        #Remove if Less than 5 strings
        df1['length'] = df1[colname].apply(lambda x: len(x.split()))
        df1=df1[df1["length"] > 5] 
        #delete column length
        df1 = df1.drop('length', 1)
        #drop ducplicates
        df1=df1.drop_duplicates(keep='first')
    
    return df1


# In[8]:


train_data = preprocess_data(df_tr,"comment_text", True)
test_data = preprocess_data(test,"comment_text", True)
train_data


# ## Vizualisation ideas

# In[9]:


df_labels=df_tr.iloc[:, 2:7]

counts = df_labels.apply(pd.value_counts)
counts = counts.T

fig = counts.plot.bar(figsize=(9,8))


# In[10]:


fig = counts.plot.bar(figsize=(9,8),y=1, legend=False, color='darkorange')


# In[11]:


def plot_top_non_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    sns.barplot(x=y[0:20],y=x[0:20])
    
plot_top_non_stopwords_barchart(train_data['comment_text'])


# In[12]:


def _get_top_ngram(corpus, n=None, w=10):
       vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
       bag_of_words = vec.transform(corpus)
       sum_words = bag_of_words.sum(axis=0) 
       words_freq = [(word, sum_words[0, idx]) 
                     for word, idx in vec.vocabulary_.items()]
       words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
       return words_freq[:w]


# In[13]:


# Top 2 consecutive non-stop words
def show(x,outfile=None, n=2,w=10 ):
    stop=set(stopwords.words('english'))

    new= x.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    top_n_bigrams=_get_top_ngram(x,n)[:w]
    x,y=map(list,zip(*top_n_bigrams))
    sns.barplot(x=y,y=x)
    if outfile is not None:
        plt.savefig(str(random.randint(0, 100000))+'_ngrams_barchart.png')


# In[14]:


show(train_data['comment_text'],outfile= not None, n=2,w=10)


# In[15]:


show(train_data['comment_text'], n=3,w=10)


# ## Model : Logistic Regression

# In[16]:


def get_topwords(logit_model, vectorizer, n_models:int = 1, n:int = 30, categorie:str = "cat") -> pd.DataFrame:
    """
    Extract top n predictors with highest coefficients from a logistic regression model and vectorizer object.
    
    Parameters
    ----------
    logit_model : LogisticRegression estimator
            A fitted LogisticRegression object from scikit-learn with coef_ attribute
    vectoriser : CountVectorizer or TfidfVectorizer
            A fitted CountVectorizer or TfidfVectorizer object from scikit-learn with get_feature_names attribute.
    n_models : int, default 17
            Indicates the number of models fitter by logit_model, i.e. n_classes.
    n : int, default 30
            The number of top predictors for each model to be returned. If None, returns all predictors
    show_idxmax : bool default True
            Indicates whether to print the keyword/predictor for each class
    Returns
    -------
    df_lambda : a pandas DataFrame object of shape (n_models,1) with a columns Keywords. Each cell in the column is
    a sorted list of tupples with top n predictors which has the form of (keyword, coefficient).
    
    """
    
    
    df_lambda = pd.DataFrame(logit_model.coef_,
                         columns = vectorizer.get_feature_names(),
                         index = [categorie for x in range(1,n_models+1)]).round(3)
    
    
    df_lambda = pd.DataFrame([df_lambda.to_dict(orient = 'index')])
    df_lambda = df_lambda.T.rename({0:'Keywords'}, axis = 1)
    
   
    falpha = lambda alpha: sorted(alpha.items(), key=lambda x:x[1], reverse=True)[:n]
    df_lambda['Keywords'] = df_lambda['Keywords'].apply(falpha)
    return df_lambda


# In[17]:


categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data.comment_text)
X_test = vectorizer.transform(test_data.comment_text)
Y_train = train_data[categories]
Y_test = test_labels.astype('object')


# In[18]:


def logreg(X_train, X_test, Y_train, Y_test, categories = categories, c=0.5, classweight=None, test_data = test_data):    
    
    evaluation = pd.DataFrame()
    evaluation['text'] = test_data['comment_text']
    
    error =[]
    
    for category in categories:
        #build classifier
        LogR = LogisticRegression(solver='saga', n_jobs=-1, C=c,class_weight = classweight)

        #re-learn & predict
        LogR.fit(X_train, Y_train[category])  


        evaluation[category] = LogR.predict(X_test) #predict
        error+=[metrics.mean_squared_log_error(Y_test[category], evaluation[category]).round(4)]
        df_lambda = get_topwords(LogR, vectorizer, n = 7, n_models = 1,categorie=str(category))
        print(df_lambda)
        print(df_lambda.Keywords.iloc[0])
    return evaluation, error



# In[19]:


def analyse(error, categories = categories, c = 0.5, classweight = None, counts_list = counts_list):
        
    data = list(zip(categories, error, [1-err for err in error]))
    col_names = ['Category', 'Error', 'Score' ]
    print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
    
    print("\033[1m" + '\nBest predicted class: ' + "\033[0m", categories[error.index(min(error))])
    print("\033[1m" + 'Worst predicted class: ' + "\033[0m" , categories[error.index(max(error))])
    

    sorted_categories = categories
    sorted_error = error
    sorted_error, sorted_categories = (list(t) for t in zip(*sorted(zip(sorted_error, sorted_categories))))

    print("\033[1m" + '\nBest to worst predictions:' + "\033[0m" , sorted_categories)


    sorted_categories = categories
    sorted_counts = counts_list
    sorted_counts, sorted_categories = (list(t) for t in zip(*sorted(zip(sorted_counts, sorted_categories))))

    print("\033[1m" + 'Least to most samples:' + "\033[0m", sorted_categories)

    #How does the error correlate to the number of samples?

    sorted_counts = counts_list
    sorted_error = error
    sorted_error, sorted_counts = (list(t) for t in zip(*sorted(zip(sorted_error, sorted_counts))))

    #print(sorted_counts, '\n', sorted_error)

    var1 = pd.Series(sorted_counts)
    var2 = pd.Series(sorted_error)

    correlation = var2.corr(var1)  
    print("\033[1m" + '\nCorrelation between sample size and error:' + "\033[0m",correlation)

    plt.scatter(var1, var2) 
    plt.xlabel('Samples')
    plt.ylabel('Error')
    plt.plot(np.unique(var1), np.poly1d(np.polyfit(var1, var2, 1))(np.unique (var1)), color = 'green');
    


# In[20]:


predictions, error = logreg(X_train, X_test, Y_train, Y_test)
analyse(error)


# In[21]:


predictions, error = logreg(X_train, X_test, Y_train, Y_test, classweight = 'balanced')
analyse(error)


# In[22]:


def roc_auc(X_train, X_test, Y_train, Y_test, categories = categories, c=0.5, classweight = None):
    
    fig = plt.figure(figsize=(10 ,6))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    
    axes = [ax1,ax2,ax3, ax4, ax5,ax6]
    fig.suptitle('ROC')

    aucs = []
    
    
    for i,category in enumerate(categories):
        
        ax = axes[i]
        
        Y_tr = Y_train[category]
        Y_ts = Y_test[category].to_numpy().astype(int)

        LogR = LogisticRegression(solver='saga', n_jobs=-1, C=c, class_weight = classweight).fit(X_train, Y_tr)

        probs = LogR.predict_proba(X_test)
        probs = probs[:, 1]
        noskill_probs = [0 for _ in range(Y_ts.shape[0])]
        
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(Y_ts, noskill_probs)
        lr_fpr, lr_tpr, _ = roc_curve(Y_ts, probs)
        
        # calculate roc auc
        ns_auc = auc(ns_fpr, ns_tpr)
        lr_auc = auc(lr_fpr, lr_tpr)
        
        aucs+=  [lr_auc.round(4)]
        
        # plot the roc curve for the model
        
        ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        ax.plot(lr_fpr, lr_tpr, lw = 2, marker = '.', label="Log (area = %0.2f)" % aucs[i]) 
        ax.legend(loc="lower right")

        # axis labels
        ax.set_title(category)
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    
    
    
    for ax in fig.get_axes():
        ax.label_outer()

    # show the legend
    #fig.legend()
        # show the plot
    print('No Skill: ROC AUC=%.4f' % (ns_auc))
    print('Logistic (mean): ROC AUC=%.4f' % (sum(aucs)/len(aucs)))
    
    data = list(zip(categories, aucs))
    col_names = ['Category', 'ROC AUC' ]
    print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
    

       
    
    fig.show()
    


# In[23]:


roc_auc(X_train, X_test, Y_train, Y_test, classweight = 'balanced')

