import sys
import os
import tensorflow as tf
#from base_network import network,Settings
import network
import numpy as np
from scipy import spatial

class environment(object):
    """
    one episode: one new environment: one entity pair-class label
    """
    def __init__(self,sentence,pos1,pos2,ind,en,y,session,model):
        #self.model
        self.sentence_set=sentence
        self.pos1=pos1
        self.pos2=pos2
        self.input_y = np.reshape(y,(1,53))
        self.select_flag=[0 for i in range(len(sentence))]
        self.select_flag[0]=1 # calculate final reward 
        self.sen2ind=ind  
        self.pernalized_step=0.0
        self.en=np.reshape(en,(-1))
        self.max_step=len(sentence)
        self.step_num= 0 #defaut select first sentence
        pathname = "./model/ATT_GRU_model-"
        test_settings = network.Settings()
        test_settings.vocab_size = 114044
        test_settings.num_classes = 53
        test_settings.big_num = 1
        self.sess=session
        self.mtest=model
        self.observation=self.first_observation()

    def step(self,action_v):
        action = np.argmax(action_v)
        self.step_num=self.step_num+1
        if action == 0 :
            if self.step_num<self.max_step:
                self.observation=self.get_newobservation()
                return self.observation , -self.pernalized_step, False,\
                      "not choose"
            if self.step_num==self.max_step:
                return self.observation , -self.pernalized_step+self.get_final_reward(), True, \
                      "not choose and episode ended"

        if action == 1 :
            self.select_flag[self.step_num-1]=1
            if self.step_num==self.max_step:
                return self.observation , -self.pernalized_step+self.get_final_reward(),\
                       True, "choose and episode ended"
            else:
                self.observation=self.get_newobservation()
                return self.observation , -self.pernalized_step,\
                       False, "choose"


    def get_final_reward(self):
        """
        sentence encoding for chosen sentences || candidate sentence encoding
        """
        selected_sen=[]
        selected_pos1=[]
        selected_pos2=[]
        for i in range(len(self.sentence_set)):
            if self.select_flag[i]==1:
                selected_sen.append(self.sentence_set[i])
                selected_pos1.append(self.pos1[i])
                selected_pos2.append(self.pos2[i])
        shape=len(selected_sen)
        #print shape
        feed_dict={}
        word_shape=[]
        word_shape.extend([0,shape])

        total_shape=np.array(word_shape)
        total_word = np.array(selected_sen)
        total_pos1 = np.array(selected_pos1)
        total_pos2 = np.array(selected_pos2)
        #input_y=[[0 for i in range(53)] for j in range(len(total_shape)-1)]
        #print np.shape(total_shape)
        #print np.shape(total_word)
        #print np.shape(total_pos1)
        #print np.shape(total_pos2)
        #print np.shape(input_y)
        feed_dict[self.mtest.total_shape] = total_shape
        feed_dict[self.mtest.input_word] = total_word
        feed_dict[self.mtest.input_pos1] = total_pos1
        feed_dict[self.mtest.input_pos2] = total_pos2
        feed_dict[self.mtest.input_y] = self.input_y

        total_loss = self.sess.run(
        [self.mtest.loglikelihood], feed_dict)
        #print total_loss
        return total_loss[0]
        
    def get_newobservation(self):
        """
        sentence encoding for chosen sentences || candidate sentence encoding 
        """ 
        selected_sen=[]
        selected_pos1=[]
        selected_pos2=[]
        for i in range(len(self.sentence_set)):
            if self.select_flag[i]==1:
                selected_sen.append(self.sentence_set[i])
                selected_pos1.append(self.pos1[i])
                selected_pos2.append(self.pos2[i])
        def get_rep(word_batch,pos1_batch,pos2_batch,shape):
            feed_dict={}
            word_shape=[]
            word_shape.extend([0,shape])
            
            total_shape=np.array(word_shape)
            total_word = np.array(word_batch)
            total_pos1 = np.array(pos1_batch)
            total_pos2 = np.array(pos2_batch)
            input_y=[[0 for i in range(53)] for j in range(len(total_shape)-1)]
            #print np.shape(total_shape)
            #print np.shape(total_word)
            #print np.shape(total_pos1)
            #print np.shape(total_pos2)
            #print np.shape(input_y)
            feed_dict[self.mtest.total_shape] = total_shape
            feed_dict[self.mtest.input_word] = total_word
            feed_dict[self.mtest.input_pos1] = total_pos1
            feed_dict[self.mtest.input_pos2] = total_pos2
            feed_dict[self.mtest.input_y] = input_y
        
            repre,max_prob,class_entropy = self.sess.run(
            [self.mtest.sen_s,self.mtest.max_prob,self.mtest.class_entropy], feed_dict)
            return repre,max_prob,class_entropy 
        shape=len(selected_sen)  
        if shape != 0: 
            old_s,old_max_prob,old_class_entropy=get_rep(selected_sen,selected_pos1,selected_pos2,shape) 
            old_sen_rep=np.reshape(old_s,(-1))
        else :
            old_s=np.random.rand(230)
            old_max_prob= np.random.random_sample()
            old_class_entropy = np.random.random_sample()
        can_s,can_max_prob,can_class_entropy=get_rep([self.sentence_set[self.step_num]],[self.pos1[self.step_num]],[self.pos2[self.step_num]],1)
        can_rep=np.reshape(can_s,(-1))
        neiji_sum=0
        max_neiji=0
        for i in range(len(selected_sen)):
            sen_enc,_,_=get_rep([selected_sen[i]],[selected_pos1[i]],[selected_pos2[i]],1)
            neiji_i=1- spatial.distance.cosine(sen_enc,can_rep)
            neiji_sum+=neiji_i
            if neiji_i>max_neiji:
                max_neiji=neiji_i
        #print [max_neiji,neiji_sum/len(selected_sen),old_max_prob,old_class_entropy]
        #print np.shape(np.concatenate((can_rep,np.array([max_neiji,neiji_sum/len(selected_sen),old_max_prob,old_class_entropy])),axis=0))
        #return np.concatenate((self.en,can_rep,np.array([max_neiji,neiji_sum/len(selected_sen),old_max_prob,old_class_entropy])),axis=0) 
        return np.concatenate((self.en,can_rep),axis=0)

    def first_observation(self):
        return self.get_newobservation()


class PR_environment(environment):
    def __init__(self,sentence,pos1,pos2,ind,en,y,session,model):
        super(PR_environment, self).__init__(sentence,pos1,pos2,ind,en,y,session,model)

    def step(self,action_v):
        action = np.argmax(action_v)
        self.step_num=self.step_num+1
        if action == 0 :
            if self.step_num<self.max_step:
                self.observation=self.get_newPRobservation()
                return self.observation , -self.pernalized_step, False,\
                      "not choose"
            if self.step_num==self.max_step:
                return self.observation , -self.pernalized_step+self.get_final_reward(), True, \
                      "not choose and episode ended"

        if action == 1 :
            self.select_flag[self.step_num-1]=1
            if self.step_num==self.max_step:
                return self.observation , -self.pernalized_step+self.get_final_reward(),\
                       True, "choose and episode ended"
            else:
                self.observation=self.get_newPRobservation()
                return self.observation , -self.pernalized_step,\
                       False, "choose"

    def get_newPRobservation(self):
        return self.get_newobservation(), self.sen2ind[self.step_num-1][0]

    def first_observation(self):
        return self.get_newPRobservation()

 
class environment_ctl(object):
    def __init__(self,word,pos1,pos2,ind,en,y,sf):
        self.word_o = np.load(word)
        self.pos1_o = np.load(pos1)
        self.pos2_o = np.load(pos2)
        self.ind_o = np.load(ind)
        self.en_o = np.load(en)
        self.y_o = np.load(y)
        self.word = []
        self.pos1 = []
        self.pos2 = []
        self.ind = []
        self.en = []
        self.y = [] 
        self.episode_index=-1
        pathname = "./model/ATT_GRU_model-"
        self.embedding=np.load('./data/vec.npy')
        test_settings = network.Settings()
        test_settings.vocab_size = 114044
        test_settings.num_classes = 53
        test_settings.big_num = 1
        with tf.Graph().as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                with tf.variable_scope("model"):
                    self.mtest = network.GRU(is_training=False, word_embeddings = self.embedding, settings = test_settings)
                    saver=tf.train.Saver()
                    saver.restore(self.sess,pathname+str(11000))
        if sf == True:
            self.s_f()
        else :
            self.word = np.array(self.word_o)
            self.pos1 = np.array(self.pos1_o)
            self.pos2 = np.array(self.pos2_o)
            self.ind = np.array(self.ind_o)
            self.en = np.array(self.en_o)
            self.y = np.array(self.y_o)

    def s_f(self):
        print "bag number:" + str(len(self.word))
        sf_idx = np.arange(len(self.word_o))
        np.random.shuffle(sf_idx)

        for i in sf_idx:
            self.word.append(self.word_o[i])
            self.pos1.append(self.pos1_o[i])
            self.pos2.append(self.pos2_o[i])
            self.ind.append(self.ind_o[i])
            self.en.append(self.en_o[i])
            self.y.append(self.y_o[i])
             
        self.word = np.array(self.word)
        self.pos1 = np.array(self.pos1)
        self.pos2 = np.array(self.pos2)
        self.ind = np.array(self.ind)
        self.en = np.array(self.en)
        self.y = np.array(self.y)

    def get_new_environment(self):
        self.episode_index=self.episode_index+1
        #while self.episode_index<len(self.word) and len(self.word[self.episode_index])<3:
        #    self.episode_index=self.episode_index+1
        if self.episode_index<len(self.word):
            return environment (self.word[self.episode_index],self.pos1[self.episode_index],self.pos2[self.episode_index],self.ind[self.episode_index],self.en[self.episode_index],self.y[self.episode_index],self.sess,self.mtest)
        else:
            return None
 
    def get_new_PR_environment(self):
        self.episode_index=self.episode_index+1
        #while self.episode_index<len(self.word) and len(self.word[self.episode_index])<3:
        #    self.episode_index=self.episode_index+1
        if self.episode_index<len(self.y):
            return PR_environment (self.word[self.episode_index],self.pos1[self.episode_index],self.pos2[self.episode_index],self.ind[self.episode_index],self.en[self.episode_index],self.y[self.episode_index],self.sess,self.mtest)
        else:
            return None 
