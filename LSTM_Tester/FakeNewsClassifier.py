import re
import os
import nltk
import joblib
import gensim
import itertools
import statistics
import numpy as np
import pandas as pd
#import seaborn as sns
import tensorflow as tf
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
from tensorflow import keras
from nltk.corpus import stopwords
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from keras.layers import SpatialDropout1D
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def testCSV_generator():

    preprocessed_Input = []
    # Loading Training Dataset
    trainingDataset = pd.read_csv("Datasets/train.csv",encoding='latin-1')
    # Droping the Nan rows
    trainingDataset = trainingDataset.dropna(subset=['text'])
    # Dividing the training dataset into 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(trainingDataset, trainingDataset['label'], test_size=0.20, random_state=42)
    # Sending text to be pre-processed for model testing
    # To display the number of row being pre-processed
    counter=1
    for text in X_test['text']:
        preprocessed_Input.append(preprocessor(text,counter))
        counter=counter+1
    # Creating a testingCSV to avoid preprocessing in case testing results are needed again
    testingCSV = pd.DataFrame(0,index=np.arange(4153),columns=['id', 'text', 'label'])
    testingCSV['text']=preprocessed_Input
    testingCSV['id']=X_test['id'].values[0:4153]
    testingCSV['label']=y_test.values[0:4153]
    # Saving pre-processed test data in CSV
    testingCSV.to_csv("Datasets/testingDataset.csv")

def trainCSV_generator():
    
    trainDataset = pd.read_csv("Datasets/train.csv",encoding='latin-1')
  
    # Real News Dataset 
    # Removing ID Column
    trainDataset=trainDataset.drop(["id"], axis=1)
    # Removing Author and Title Column
    trainDataset=trainDataset.drop(["author","title"], axis=1)
    # Extracted Real News Data
    real_data= pd.read_csv("Datasets/REAL_DATA.csv",encoding='latin-1')
    # Removing unused columns
    real_data=real_data.drop(["Author","Title","Source","Link","DateOfExtraction","DateOfPublication"], axis=1)
    # Removing unnamed column
    real_data=real_data.drop(real_data.columns[0],axis=1)
    # Adding a label column in real dataset
    real_data['label']=0
    # Changing column name
    real_data = real_data.rename(columns={'Article': 'Text'})
    # Converting column names into lowercase
    real_data.columns= real_data.columns.str.lower()

    # Fake News Dataset 
    # Extracted Fake News Data
    fake_data= pd.read_csv("Datasets/Fake.csv",encoding='latin-1')
    # Removing unused columns
    fake_data=fake_data.drop(["title","subject","date"], axis=1)
    # Adding a label column in real dataset
    fake_data['label']=1
    # Converting column names into lowercase
    fake_data.columns= real_data.columns.str.lower()
    # Only using 5000 articles
    fake_data=fake_data.head(5000)

    #Appending datasets
    train_set=pd.concat([trainDataset,real_data,fake_data])
    # Adding ID Column
    train_set.insert(0, 'id', range(0, 0 + len(train_set)))
    # Saving new training data in CSV
    train_set.to_csv("Datasets/trainingDataset.csv")

def title_trainCSV_generator():
    
    trainDataset = pd.read_csv("Datasets/train.csv",encoding='latin-1')
  
    # Real News Dataset 
    # Removing ID Column
    trainDataset=trainDataset.drop(["id"], axis=1)
    # Removing Author and Title Column
    trainDataset=trainDataset.drop(["author"], axis=1)
    # Extracted Real News Data
    real_data= pd.read_csv("Datasets/REAL_DATA.csv",encoding='latin-1')
    # Removing unused columns
    real_data=real_data.drop(["Author","Source","Link","DateOfExtraction","DateOfPublication"], axis=1)
    # Removing unnamed column
    real_data=real_data.drop(real_data.columns[0],axis=1)
    # Adding a label column in real dataset
    real_data['label']=0
    # Changing column name
    real_data = real_data.rename(columns={'Article': 'Text'})
    # Converting column names into lowercase
    real_data.columns= real_data.columns.str.lower()

    # Fake News Dataset 
    # Extracted Fake News Data
    fake_data= pd.read_csv("Datasets/Fake.csv",encoding='latin-1')
    # Removing unused columns
    fake_data=fake_data.drop(["subject","date"], axis=1)
    # Adding a label column in real dataset
    fake_data['label']=1
    # Converting column names into lowercase
    fake_data.columns= real_data.columns.str.lower()
    # Only using 5000 articles
    fake_data=fake_data.head(5000)

    #Appending datasets
    train_set=pd.concat([trainDataset,real_data,fake_data])
    # Adding ID Column
    train_set.insert(0, 'id', range(0, 0 + len(train_set)))
    # Saving new training data in CSV
    train_set.to_csv("Datasets/titletrainingDataset.csv")
######################################## Data Pre-Processing ######################################## 
def preprocessor(inputText,counter):

    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    review = re.sub('[^a-zA-Z]', ' ',inputText)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    print(counter)
    return review

def text_preprocessor():
    trainingDataset = pd.read_csv("Datasets/trainingDataset.csv",encoding='latin-1')
    # Sending text to be pre-processed for model training
    # Droping the Nan rows
    trainingDataset = trainingDataset.dropna(subset=['text'])
    # To display the number of row being pre-processed
    preprocessed_Input = []
    counter=1
    for text in trainingDataset['text']:
        preprocessed_Input.append(preprocessor(text,counter))
        counter=counter+1
    # Creating a preprocessed_trainingCSV to avoid preprocessing
    trainingDataset['text']=preprocessed_Input
    # Saving pre-processed test data in CSV
    trainingDataset.to_csv("Datasets/preprocessed_trainingData.csv")

def title_preprocessor():
    trainingDataset = pd.read_csv("Datasets/titletrainingDataset.csv",encoding='latin-1')
    # Sending text to be pre-processed for model training
    # Droping the Nan rows
    trainingDataset = trainingDataset.dropna(subset=['title'])
    # To display the number of row being pre-processed
    preprocessed_Input = []
    counter=1
    for text in trainingDataset['title']:
        preprocessed_Input.append(preprocessor(text,counter))
        counter=counter+1
    # Creating a preprocessed_trainingCSV to avoid preprocessing
    trainingDataset['title']=preprocessed_Input
    # Saving pre-processed test data in CSV
    trainingDataset.to_csv("Datasets/preprocessed_titletrainingData.csv")
######################################## Data Pre-Processing ######################################### 

######################################## Feature Extraction ######################################## 
def feature_extractor(maxlen,EMBEDDING_DIM):
    trainingDataset= pd.read_csv("Datasets/preprocessed_trainingData.csv",encoding='latin-1')
    # Droping the Nan rows
    trainingDataset = trainingDataset.dropna(subset=['text'])
    X_final = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    for par in trainingDataset["text"].values:
        tmp = []
        sentences = nltk.sent_tokenize(par)
        for sent in sentences:
            sent = sent.lower()
            tokens = tokenizer.tokenize(sent)
            filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
            tmp.extend(filtered_words)
        X_final.append(tmp)
    w2v_model = gensim.models.Word2Vec(sentences=X_final, size=EMBEDDING_DIM, window=5, min_count=1)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_final)
    X_final = tokenizer.texts_to_sequences(X_final)
    word_index = tokenizer.word_index
    X_final = pad_sequences(X_final,maxlen=maxlen)
    y_final = np.array(trainingDataset['label'])  
    vocab_size = len(tokenizer.word_index) + 1
    vocab = len(word_index) + 1
    embedding_vectors  = np.zeros((vocab, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vectors [i] = w2v_model[word]
    return [embedding_vectors,vocab_size]
######################################## Feature Extraction ######################################### 

######################################## Model Training ########################################
def text_modeltraining(): 
    trainingDataset= pd.read_csv("Datasets/preprocessed_trainingData.csv",encoding='latin-1')
    lstm_out=200
    voc_size=5000
    #################### Feature Extraction ####################
    maxlen=1000
    EMBEDDING_DIM = 100
    extracted_features=feature_extractor(maxlen,EMBEDDING_DIM)
    #################### Model Architecture  ####################
    model = Sequential()
    model.add(Embedding(extracted_features[1], output_dim=EMBEDDING_DIM, weights=[extracted_features[0]], input_length=maxlen, trainable=False))
    model.add(LSTM(units=128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    #################### Model Training  ####################
    X_train,X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.20, random_state = 0)
    model.fit(X_train, y_train, epochs = 10, batch_size=128, verbose = 1)
    # save model and architecture to single file
    model.save("Model/FakeNewsLSTM.h5")
    print("Saved model to disk")

def title_modeltraining(): 
    trainingDataset= pd.read_csv("Datasets/preprocessed_titletrainingData.csv",encoding='latin-1')
    voc_size=5000
    #################### Feature Extraction ####################
    # Droping the Nan rows
    trainingDataset = trainingDataset.dropna(subset=['title'])
    onehot_repr = [one_hot(words, voc_size) for words in trainingDataset['title']]
    sent_length = 25
    embedded_doc = pad_sequences(onehot_repr,padding = 'pre', maxlen= sent_length)  
    X_final = np.array(embedded_doc)
    y_final = np.array(trainingDataset['label'])
    embedding_vector_features = 40
    #################### Model Architecture  ####################
    model = Sequential()
    model.add(Embedding(voc_size,embedding_vector_features, input_length = sent_length))
    model.add(Dropout(0.3))
    model.add(LSTM(200))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    #################### Model Training  ####################
    X_train,X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.20, random_state = 0)
    model.fit(X_train, y_train, epochs = 200, batch_size=64, verbose = 1)
    # save model and architecture to single file
    model.save("Model/TitleFakeNewsLSTM.h5")
    print("Saved model to disk")
######################################## Model Training ########################################

######################################## Model Testing ########################################
def text_predict(): 
    model = tf.keras.models.load_model('Model/FakeNewsLSTM.h5')
    os.system('cls')
    testDataset = pd.read_csv("Datasets/preprocessed_trainingData.csv",encoding='latin-1')
    # Droping the Nan rows
    testDataset = testDataset.dropna(subset=['text'])
    X_final = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    for par in testDataset["text"].values:
        tmp = []
        sentences = nltk.sent_tokenize(par)
        for sent in sentences:
            sent = sent.lower()
            tokens = tokenizer.tokenize(sent)
            filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
            tmp.extend(filtered_words)
        X_final.append(tmp)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_final)
    #joblib.dump(tokenizer, 'Model/LSTM_tokenizer.pkl')
    #joblib.dump(X_final, 'Model/X_final.pkl')
    X_final = tokenizer.texts_to_sequences(X_final)
    X_final = pad_sequences(X_final,maxlen=1000)
    y_final = np.array(testDataset['label'])  
    vocab_size = len(tokenizer.word_index) + 1
    X_train,X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.20, random_state = 0)
    prediction=(model.predict(X_test) > 0.5).astype("int32")
    # Classification report
    report = classification_report(y_test, prediction)
    cm = confusion_matrix(y_test, prediction)
    # Confustion Matrix
    os.system('cls')
    print(report)
    print(cm)



def input_predict(text,model):      
    #trainDataset = pd.read_csv("Datasets/preprocessed_trainingData.csv",encoding='latin-1')
    #trainDataset = trainDataset.dropna(subset=['text'])
    
    inputValue_test = []

    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ',text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    sentences = nltk.sent_tokenize(review)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        filtered_words = ' '.join(filtered_words)
        inputValue_test.append(filtered_words)

    voc_len=len(inputValue_test)
    
    
    #tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(trainDataset['text'].values)
    #joblib.dump(tokenizer, 'Model/LSTM_Tokenizer.pkl')
    
    tokenizer = joblib.load('Model/LSTM_Tokenizer.pkl')
    X_test = tokenizer.texts_to_sequences(inputValue_test)
    padded = pad_sequences(X_test, maxlen=1000)
    y_pred=(model.predict(padded) > 0.5).astype("int32")
    return(y_pred[0][0])
    
    #onehot_repr_test = [one_hot(word, voc_len) for word in inputValue_test]
    #embedded_docs_test=pad_sequences(onehot_repr_test,maxlen=1000)
    #X_test=np.array(embedded_docs_test)
    #y_pred=(model.predict(X_test) > 0.5).astype("int32")
    #print(y_pred[0][0])


def title_predict(): 
    model = tf.keras.models.load_model('Model/TitleFakeNewsLSTM.h5')
    os.system('cls')
    testDataset = pd.read_csv("Datasets/preprocessed_titletrainingData.csv",encoding='latin-1')
    # Droping the Nan rows
    testDataset = testDataset.dropna(subset=['title'])
    onehot_repr_test = [one_hot(word, 5000) for word in testDataset['title']]
    embedded_docs_test=pad_sequences(onehot_repr_test,padding='pre',maxlen=25)
    X_train,X_test, y_train, y_test = train_test_split(embedded_docs_test, testDataset['label'], test_size = 0.20, random_state = 0)
    y_pred=(model.predict(X_test) > 0.5).astype("int32")
    y_pred=np.concatenate(y_pred, axis=0 )
    unique, counts = np.unique(y_pred, return_counts=True)
    # Classification report
    report = classification_report(y_test, y_pred)
    os.system('cls')
    print(model.summary())
    print("Number of 0's and 1's predicted by model")
    print(np.asarray((unique, counts)).T)
    print("\nModel Classification Report\n")
    print(report)
######################################## Model Testing ########################################


######################################## Main Function ########################################
if __name__ == "__main__":
    # Call the function below to generate Text trainDataset.csv file to be used for training
    #trainCSV_generator()
        
    # Call the function below to generate Title trainDataset.csv file to be used for training
    #title_trainCSV_generator()
    os.system('cls')
    # Taking input from user to train a model
    choice = input(" Press 1 for Data Pre-Processing  \n Press 2 for Model Training \n Press 3 for Model Testing \n Press 4 for Live Script Testing \n Enter Choice: ")
    ####################### Data Pre-Processing ####################### 
    if choice=='1':    
        os.system('cls')
        choice = input(" Press 1 for Text Model Pre-Processing  \n Press 2 for Text Model Pre-Processing \n Enter Choice: ")
        if choice=='1': 
            ########### For Text Model  ###########
            text_preprocessor()
        elif choice=='2':  
            ############ For Title Model  ###########
            title_preprocessor()
    ####################### Data Pre-Processing ####################### 
    
    ####################### Model Training #######################
    elif choice=='2':  
        os.system('cls')
        choice = input(" Press 1 for Text Model Training  \n Press 2 for Title Model Training \n Enter Choice: ")
        if choice=='1': 
            ########### For Text Model  ###########
            text_modeltraining()
        elif choice=='2':  
            ############ For Title Model  ###########
            title_modeltraining()
    ####################### Model Training #######################
    elif choice=='3':  
        os.system('cls')    
        choice = input(" Press 1 for Text Model Testing  \n Press 2 for Title Model Testing \n Enter Choice: ")
        if choice=='1': 
            ########### For Text Model  ###########
            text_predict()
        elif choice=='2':  
            ############ For Title Model  ###########
            title_predict()
    ####################### Live Testing #######################
    elif choice=='4': 
        model = tf.keras.models.load_model('Model/FakeNewsLSTM.h5')
        os.system('cls')    
        trainDataset = pd.read_csv("Datasets/preprocessed_trainingData.csv",encoding='latin-1')
        print("Testing:\n ")
        # For 1st Thousand articles
        for x in range(2):
            text=trainDataset["text"][x]
            print("\nid: ",trainDataset['id'][x], end ="")
            print(" Actual: ",trainDataset['label'][x], end ="")
            print(" Predicted: ",input_predict(text,model))

    os.system("pause")
 ######################################## Main Function ########################################







