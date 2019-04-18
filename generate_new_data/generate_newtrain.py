import os
import sys
#import tensorflow as tf
import numpy as np

def generate_new_train(select_action):
    train_word=np.load('./data/train_word.npy')
    train_pos1=np.load('./data/train_pos1.npy')
    train_pos2=np.load('./data/train_pos2.npy')
    train_y=np.load('./data/train_y.npy')
    train_word_new = []
    train_pos1_new = []
    train_pos2_new = []
    train_y_new = []
    index=0
    #print np.shape(select_action),np.shape(train_word),np.shape(testall_pos1)
    for i in range(len(train_word)):
        select_word=[]
        select_pos1=[]
        select_pos2=[]
        #select_y=[]
        if 0 == np.sum(select_action[i]):
            index = index + 1 
            continue
        else:
            print i,index,len(train_word[i]),len(select_action[index])
            assert len(train_word[i])==len(select_action[index])
            for j in range(len(select_action[index])):
                if select_action[index][j]==1:
                    select_word.append(train_word[i][j])
                    select_pos1.append(train_pos1[i][j])
                    select_pos2.append(train_pos2[i][j])
                   #select_y.append(select_y[i][j])
            index=index+1    
            train_word_new.append(select_word) 
            train_pos1_new.append(select_pos1)
            train_pos2_new.append(select_pos2)
            train_y_new.append(train_y[i])
    train_word_new = np.array(train_word_new)
    train_pos1_new = np.array(train_pos1_new)
    train_pos2_new = np.array(train_pos2_new)
    train_y_new = np.array(train_y_new)
    np.save("./data/train_word_new.npy",train_word_new)
    np.save("./data/train_pos1_new.npy",train_pos1_new)
    np.save("./data/train_pos2_new.npy",train_pos2_new)
    np.save("./data/train_y_new.npy",train_y_new)

if __name__=="__main__":
    pass
    #s_t=np.load("../data/select_train.npy")
    #generate_new_train(s_t)
