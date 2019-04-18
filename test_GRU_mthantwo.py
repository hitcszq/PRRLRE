import tensorflow as tf
import numpy as np
import time
import datetime
import os
from base_network import GRU,Settings
from sklearn.metrics import average_precision_score

FLAGS = tf.app.flags.FLAGS
#change the name to who you want to send
#tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')

#if you want to try itchat, please set it to True

def test_GRU(test_tuple):

    # ATTENTION: change pathname before you load your model
    pathname = "../model/ATT_GRU_model-"
    
    wordembedding = np.load('../data/vec.npy')

    test_settings = network.Settings()
    test_settings.vocab_size = 114044
    test_settings.num_classes = 53    
    test_settings.big_num = 72

    big_num_test = test_settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):
    
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                
                #print np.shape(total_shape)
                print np.shape(total_word)
                #print np.shape(total_pos1)
                #print np.shape(total_pos2)
                #print np.shape(y_batch)
                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy ,prob= sess.run(
                    [mtest.loss, mtest.accuracy,mtest.prob], feed_dict)
                return prob,accuracy
            
            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings = wordembedding, settings = test_settings)

            saver = tf.train.Saver()
            

            # ATTENTION: change the list to the iters you want to test !!
            #testlist = range(9025,14000,25)
            testlist = [11000]
            for model_iter in testlist:

                saver.restore(sess,pathname+str(model_iter))
                print("Evaluating P@N for iter "+str(model_iter))


                
                test_y = np.load("./data/testall_y.npy")
                test_word = np.load(test_tuple[0])
                test_pos1 = np.load(test_tuple[1])
                test_pos2 = np.load(test_tuple[2])
                print np.shape(test_word)
                allprob = [] 
                acc = []
                for i in range(int(len(test_word)/float(test_settings.big_num))):
                    prob,accuracy = test_step(test_word[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos1[i*test_settings.big_num:(i+1)*test_settings.big_num],test_pos2[i*test_settings.big_num:(i+1)*test_settings.big_num],test_y[i*test_settings.big_num:(i+1)*test_settings.big_num])
                    acc.append(np.mean(np.reshape(np.array(accuracy),(test_settings.big_num))))
                    prob = np.reshape(np.array(prob),(test_settings.big_num,test_settings.num_classes))
                    print np.shape(prob)
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob),(-1))
                order = np.argsort(-allprob)

                print 'saving all test result...'
                current_step = model_iter
                
                # ATTENTION: change the save path before you save your result !!
                np.save('./out/DQN_allprob_mthantwo_iter_'+str(current_step)+'.npy',allprob)
                allans = np.load('./data/allans_mthantwo.npy')
                #print np.shape(allans) 
                #caculate the pr curve area
                average_precision = average_precision_score(allans,allprob)
                print 'RL PR curve area:'+str(average_precision)

                if itchat_run:
                    itchat.send('PR curve area:'+str(average_precision),FLAGS.wechat_name)

                time_str = datetime.datetime.now().isoformat()
                print time_str


if __name__ == "__main__":
    pass
