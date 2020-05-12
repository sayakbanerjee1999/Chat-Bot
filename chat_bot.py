"""
Created on Thu May 5, 2020

@author: Sayak Banerjee
"""

import pickle
import numpy as np
import pandas as pd

#Load the texts
with open("train_qa.txt", "rb") as myfile:   # Unpickling
    train_data =  pickle.load(myfile)
    
with open("test_qa.txt", "rb") as myfile:   # Unpickling
    test_data =  pickle.load(myfile)
    
    

#Explore the format of the data
type(train_data)
type(test_data)

' '.join(train_data[0][0])                                                      # --> Statement
' '.join(train_data[0][1])                                                      # --> Question 
train_data[0][2]                                                                # --> Answer






'''Build the Vocab'''

all_data = train_data + test_data
len(all_data)                                                  

vocab = set()

for statement, question, answer in all_data:
    vocab = vocab.union(set(statement))
    vocab = vocab.union(set(question))
    
vocab.add("yes")
vocab.add("no")

len(vocab)

vocab_size = len(vocab) + 1                                                     # --> we add an extra space to hold a 0 for Keras's pad_sequences


#Find the Story with Max Length
max_story = []

for story, question, answer in all_data:
    max_story.append(len(story))
    
max_story_len = max(max_story)


#Find the Question with Max Length
max_question = []

for story, question, answer in all_data:
    max_question.append(len(question))
    
max_question_len = max(max_question)






'''Vectorize the Data'''

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters = '')
tokenizer.fit_on_texts(vocab)

tokenizer.word_index

#train_story_text = []
#train_question_text = []
#train_answer_text = []

#for story, question, answer in train_data:
#    train_story_text.append(story)
#    train_question_text.append(question)

#train_story_seq = tokenizer.texts_to_sequences(train_story_text)   


#Now Lets Create a Function that will Vectorize data for us
def vectorize(data, word_index = tokenizer.word_index, max_story_len = max_story_len, max_question_len = max_question_len):
    
    X = []                                                                      # --> For Stories
    Xq = []                                                                     # --> For Questions
    
    Y = []                                                                      # --> For Answers/Target
    
    for story, question, answer in data:
        x = []   
        xq = []
                                                               # --> For each story
        for word in story:
            x.append(word_index[word.lower()])
            
        for word in question:
            xq.append(word_index[word.lower()])
        
        y = np.zeros(len(word_index) + 1)
        
        y[word_index[answer]] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
    return (pad_sequences(X, maxlen = max_story_len), pad_sequences(Xq, maxlen = max_question_len), 
            np.array(Y))
    

inputs_train, questions_train, answers_train = vectorize(train_data)

inputs_test, questions_test, answers_test = vectorize(test_data)


sum(answers_train)
sum(answers_test)






'''Build the Model following the Paper on End-To-End Memory Networks'''

from keras.models import Sequential, Model        
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Permute, Dense, Dropout
from keras.layers import LSTM
from keras.layers import add, dot, concatenate       
        

#Create the Input Placeholder -> (Input() is used to instantiate a Keras tensor)  
input_sequence = Input((max_story_len,))                                        #The Second Element is the unknown batch size
question = Input((max_question_len,))


#Input encoder M (Memory Units xi)
# .add is a method of the Sequential Class
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_size, output_dim = 64))
input_encoder_m.add(Dropout(0.3))

# This encoder will output:- (samples, story_maxlen, embedding_dim)       


#Input encoder C
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_size,output_dim = max_question_len))
input_encoder_c.add(Dropout(0.3))

# This encoder will output: (samples, story_maxlen, query_maxlen)   


#Question Encoder     
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_size,
                               output_dim = 64,
                               input_length = max_question_len))
question_encoder.add(Dropout(0.3))


#Encode INPUT_SEQUENCE & the QUESTION
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


#Dot product between input_encoded_m and question_encoded followed by a Softmax Activation
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)


#Next we take the Weighted sum of Input_encoded_m and match (Permutes the dimensions of the input according to a given pattern)
response = add([match, input_encoded_c])                                        # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)                                            # (samples, query_maxlen, story_maxlen)


#Now we Concatenate the response with the question_encoded (Layer that concatenates a list of inputs)
answer = concatenate([response, question_encoded])
answer


#Add a LSTM(RNN) Layer
answer = LSTM(units = 32)(answer)


#Regularize with a Dropout Layer and Add a Dense Layer
#Dense Layer effective does:-  output = activation(dot(input, weights/kernels) + biases)
answer = Dropout(0.5)(answer)
answer = Dense(units = vocab_size)(answer)


#Finally Add a Softmax Activation Layer
answer = Activation('softmax')(answer)


#Build the Final Model
model = Model([input_sequence, question], answer)


#Compile the Model
model.compile( optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics= ['accuracy'])
model.summary()


#Fit the Model to the Training Data
model.fit([inputs_train, questions_train], answers_train,
          batch_size = 32,
          epochs = 120,
          verbose = 1,
          validation_data = ([inputs_test, questions_test], answers_test))


#Save the Weights 
myfilename = 'chatbot.h5'
model.save(myfilename)







'''Evaluate the Model and Predict Results'''

predictions = model.predict(([inputs_test, questions_test]))

predictions[0]

first_story = ' '.join(test_data[0][0])
first_question = ' '.join(test_data[0][1])
first_answer = test_data[0][2]

print("First Story:- " + first_story)
print("First Question:- " + first_question)
print("Actual Test Data Answer is " + first_answer.upper())

#Predicted Answer for 1st Story and Question
index_max = np.argmax(predictions[0])

predicted_word = tokenizer.index_word[index_max]

print("\n")

print("Predicted Answer:- " + predicted_word.upper()) 
print("Probability of Certainly was:- " + str(predictions[0][index_max]))



# Now Lets Check the Accuracy of our Model on the Test Set
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#We will not be comparing the Yes or No's but compate the word_index
pred_labels = []

for each_prediction in predictions:
    pred_labels.append(np.argmax(each_prediction))

df_pred_labels = pd.DataFrame(pred_labels, columns = ['Predicted_Labels'])
df_pred_labels['Predicted_Labels'].value_counts()
    

true_labels = []

for story, question, answer in test_data:
    true_labels.append(tokenizer.word_index[answer])

df_true_labels = pd.DataFrame(pred_labels, columns = ['True_Labels'])
df_true_labels['True_Labels'].value_counts()

print(confusion_matrix(true_labels, pred_labels))
print("Accuracy of the Model is:- ", accuracy_score(true_labels, pred_labels) * 100, "%")
#Accuracy is > 80% --> Pretty Good
print(classification_report(true_labels, pred_labels))






'''Evaluate the Model with Your Own Data'''

#Make Sure your Story and Question has Words from the Vocab 
my_story = "John went to the bedroom . Daniel grabbed the football dropped the football in the garden ."
my_question = "Is the football in the garden ?"
my_answer = "yes"

#Split the Data to be fed in to vectorize function
my_data = [(my_story.split(), my_question.split(), my_answer)]

my_vec_story, my_vec_question, my_vec_answer = vectorize(my_data)


my_story_pred = model.predict(([my_vec_story, my_vec_question]))

ind_max = np.argmax(my_story_pred)
my_predicted_word = tokenizer.index_word[ind_max]

print("My story:- ", my_story)
print("My Question:- ", my_question)
print("My Answer:- ", my_answer)
print("\n")
print("Predticted Answer to my Question:- " + my_predicted_word.upper())
print("Probability of Certainty of the Prediction:- ", str(my_story_pred[0][ind_max]))