# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:18:08 2020

@author: user
"""

def vectorize(data, word_index = tokenizer.word_index, max_story_len = max_story_len, max_question_len = max_question_len):
    '''
    INPUT: 
    
    data: consisting of Stories,Queries,and Answers
    word_index: word index dictionary from tokenizer
    max_story_len: the length of the longest story (used for pad_sequences function)
    max_question_len: length of the longest question (used for pad_sequences function)
    
    
    OUTPUT:
    
    Vectorizes the stories,questions, and answers into padded sequences. We first loop for every story, query , and
    answer in the data. Then we convert the raw words to an word index value. Then we append each set to their appropriate
    output list. Then once we have converted the words to numbers, we pad the sequences so they are all of equal length.
    
    Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)
    '''
    
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
