import os
import sys
import tensorflow as tf
import numpy as np

if __name__=="__main__":
    testall_word=np.load('./data/testall_word.npy')
    testall_pos1=np.load('./data/testall_pos1.npy')
    testall_pos2=np.load('./data/testall_pos2.npy')
    testall_word_new = []
    testall_pos1_new = []
    testall_pos2_new = []
    index=0
#    print np.shape(select_action),np.shape(testall_word),np.shape(testall_pos1)
    for i in range(len(testall_word)):
        select_word=[]
        select_pos1=[]
        select_pos2=[]
        if len(testall_word[i])<3:
            continue
            select_word=testall_word[i]
            select_pos1=testall_pos1[i]
            select_pos2=testall_pos2[i]
        else:
            select_word.append(testall_word[i][0])
            select_pos1.append(testall_pos1[i][0])
            select_pos2.append(testall_pos2[i][0])
            for j in range(len(testall_word[i]))[1:]:
                if np.random.uniform()<0.319846:
                    select_word.append(testall_word[i][j])
                    select_pos1.append(testall_pos1[i][j])
                    select_pos2.append(testall_pos2[i][j])
            index=index+1    
        testall_word_new.append(select_word) 
        testall_pos1_new.append(select_pos1)
        testall_pos2_new.append(select_pos2)
    np.save("./data/testall_word_random.npy",testall_word_new)
    np.save("./data/testall_pos1_random.npy",testall_pos1_new)
    np.save("./data/testall_pos2_random.npy",testall_pos2_new)
