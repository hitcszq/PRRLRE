import os
import sys
import tensorflow as tf
import numpy as np

def generate_new_test(select_action):
    testall_word=np.load('./data/testall_word.npy')
    testall_pos1=np.load('./data/testall_pos1.npy')
    testall_pos2=np.load('./data/testall_pos2.npy')
    testall_y=np.load('./data/testall_y.npy')
    testall_word_new = []
    testall_pos1_new = []
    testall_pos2_new = []
    testall_y_new = []
    index=0
    #print np.shape(select_action),np.shape(testall_word),np.shape(testall_pos1)
    for i in range(len(testall_word)):
        select_word=[]
        select_pos1=[]
        select_pos2=[]
        if 0 == np.sum(select_action[i],axis=0):
            index = index + 1
            continue
            #select_word=testall_word[i]
            #select_pos1=testall_pos1[i]
            #select_pos2=testall_pos2[i]
        else:
            #print i,testall_word[i],select_action[index],len(testall_word[i]),len(select_action[index])
            assert len(testall_word[i])==len(select_action[index])
            for j in range(len(select_action[index])):
                if select_action[index][j]==1:
                    select_word.append(testall_word[i][j])
                    select_pos1.append(testall_pos1[i][j])
                    select_pos2.append(testall_pos2[i][j])
            index=index+1    
            testall_word_new.append(select_word) 
            testall_pos1_new.append(select_pos1)
            testall_pos2_new.append(select_pos2)
            testall_y_new.append(testall_y[i])
    test_word_new = np.array(testall_word_new)
    test_pos1_new = np.array(testall_pos1_new)
    test_pos2_new = np.array(testall_pos2_new)
    test_y_new=np.array(testall_y_new)
    np.save("./data/test_word_new.npy",test_word_new)
    np.save("./data/test_pos1_new.npy",test_pos1_new)
    np.save("./data/test_pos2_new.npy",test_pos2_new)
    np.save("./data/test_y_new.npy",test_y_new)
if __name__=="__main__":
    pass
