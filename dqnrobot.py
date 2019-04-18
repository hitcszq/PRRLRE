import tensorflow as tf
import numpy as np
import random
from collections import deque
from fol import FOL_But

# Hyper Parameters:
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 32.  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 1000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
FINAL_EPSILON = 0
INITIAL_EPSILON = 0
# or alternative:
# FINAL_EPSILON = 0.0001  # final value of epsilon
# INITIAL_EPSILON = 0.01  # starting value of epsilon
UPDATE_TIME = 100
EXPLORE = 100000.  # frames over which to anneal epsilon


class RobotCNNDQN(object):

    def __init__(self, actions=2, vocab_size=20000, max_len=120, embeddings=[]):
        print("Creating a robot: CNN-DQN")
        # replay memory
        self.replay_memory = deque()
        self.time_step = 0
        self.action = actions
        self.w_embeddings = embeddings
        self.max_len = max_len
        self.num_classes = 5
        self.epsilon = INITIAL_EPSILON

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_size = 40
        self.create_qnetwork()
        self.saver = tf.train.Saver()

    def initialise(self, max_len, embeddings):
        self.max_len = max_len
        self.w_embeddings = embeddings
        self.vocab_size = len(self.w_embeddings)
        self.embedding_size = len(self.w_embeddings[0])
        self.create_qnetwork()
        self.saver = tf.train.Saver()

    def create_qnetwork(self):
        # read a sentence

        self.state = tf.placeholder(
            tf.float32, [None, 330], name="input_state")
        # network weights
        # size of a sentence = 384
        self.w_fc1 = self.weight_variable([330, 256])
        self.b_fc1 = self.bias_variable([256])
        self.w_fc2 = self.weight_variable([256, self.action])
        self.b_fc2 = self.bias_variable([self.action])

        # hidden layers
        self.h_fc1_all = tf.nn.relu(tf.matmul(self.state, self.w_fc1) + self.b_fc1)
        # Q Value layer
        self.qvalue = tf.matmul(self.h_fc1_all, self.w_fc2) + self.b_fc2
        # action input
        self.action_input = tf.placeholder("float", [None, self.action])
        # reword input
        self.y_input = tf.placeholder("float", [None])

        self.q_distr = tf.placeholder("float", [None, self.action])

        q_action = tf.reduce_sum(
            tf.multiply(self.qvalue, self.action_input), reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.y_input - q_action))
        # error function
        self.cost_t = tf.reduce_mean(tf.square(self.q_distr[0] - self.qvalue[0]))

        self.train_q=tf.train.AdamOptimizer(1e-6).minimize(self.cost_t)
        # train method
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        self.sess = tf.Session()
        # ? multiple graphs: how to initialise variables ?
        self.sess.run(tf.global_variables_initializer())

    def train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        next_state_sent_batch=[]
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        for state in next_state_batch:
            next_state_sent_batch.append(state)

        # Step 2: calculate y
        y_batch = []
        qvalue_batch = self.sess.run(self.qvalue, feed_dict={
                                     self.state: next_state_sent_batch})
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.max(qvalue_batch[i]))
        sent_batch = []
        confidence_batch = []
        predictions_batch = []
        for state in state_batch:
            sent_batch.append(state)

        self.sess.run(self.trainStep, feed_dict={
                      self.y_input: y_batch, self.action_input: action_batch, self.state: sent_batch})

        # save network every 10000 iteration
        # if self.time_step % 10000 == 0:
        #    self.saver.save(self.sess, './' +
        #                    'network' + '-dqn', global_step=self.time_step)

    def update(self, observation, action, reward, observation2, terminal):
        self.current_state = observation
        #newState = observation2
        new_state = observation2
        self.replay_memory.append(
            (self.current_state, action, reward, new_state, terminal))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()
        global OBSERVE
        if self.time_step > OBSERVE:
            # Train the network
            self.train_qnetwork()

        self.current_state = new_state
        self.time_step += 1

    def get_action(self, observation):
        #print "DQN is smart."
        self.current_state = observation
        qvalue = self.sess.run(self.qvalue, feed_dict={self.state: [ self.current_state ]})[0]

        action = np.zeros(self.action)
        action_index = 0
        # if self.timeStep % FRAME_PER_ACTION == 0:
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action)
            action[action_index] = 1
        else:
            action_index = np.argmax(qvalue)
            action[action_index] = 1
        # else:
        #    action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

class RobotPRDQN(RobotCNNDQN):
    
    def __init__(self, actions=2, vocab_size=20000, max_len=120, embeddings=[]):
        super(RobotPRDQN, self).__init__(actions, vocab_size, max_len, embeddings)
        
    def get_action(self, observation):
        #print "DQN is smart."
        self.current_state = observation
        qvalue = self.sess.run(self.qvalue, feed_dict={self.state: [ self.current_state ]})[0]

        action = np.zeros(self.action)
        action_index = 0
        # if self.timeStep % FRAME_PER_ACTION == 0:
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action)
            action[action_index] = 1
        else:
            action_index = np.argmax(qvalue)
            action[action_index] = 1
        # else:
        #    action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action, qvalue

    def update(self, observation, action, reward, observation2, terminal, ind, qvalue):
        self.current_state = observation
        #newState = observation2
        new_state = observation2
        self.replay_memory.append(
                (self.current_state, action, reward, new_state, terminal, ind, qvalue))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()
        global OBSERVE
        if self.time_step > OBSERVE:
            # Train the network
            self.train_qnetwork()

        self.current_state = new_state
        self.time_step += 1

    def train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        ind_batch = [data[5] for data in minibatch]
        qvalue = [data[6]  for data in minibatch]
        #print ind_batch
        #print qvalue
        nclasses = 2
        next_state_sent_batch = []
        for state in next_state_batch:
            next_state_sent_batch.append(state)

        # Step 2: calculate y
        y_batch = []
        qvalue_batch = self.sess.run(self.qvalue, feed_dict={
                                     self.state: next_state_sent_batch })
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.max(qvalue_batch[i]))
        sent_batch = []
        for state in state_batch:
            sent_batch.append(state)
        self.sess.run(self.trainStep, feed_dict={
                      self.y_input: y_batch, self.action_input: action_batch, self.state: sent_batch})
	
        qvalue = np.array(qvalue).reshape(BATCH_SIZE,2)
        ind_batch = np.array(ind_batch).reshape(BATCH_SIZE,1) 

        feature =  np.concatenate((ind_batch,qvalue),axis=1)
        self.rule = FOL_But(nclasses, state_batch, feature)
        q_y_given_x = qvalue
        print q_y_given_x
        # combine rule constraints`
        distr = self.calc_rule_constraints(self.rule)
        print distr
        q_y_given_x *= distr
        # normalize
        #print q_y_given_x
        n = len(state_batch)
        #print np.sum(q_y_given_x,axis=1)
        n_q_y_given_x = q_y_given_x / np.sum(q_y_given_x,axis=1).reshape((n,1))
        #print n_q_y_given_x
        self.q_y_given_x = n_q_y_given_x
	#print "PR train"
	self.sess.run(self.train_q, feed_dict={
                      self.q_distr: self.q_y_given_x, self.action_input: action_batch, self.state: sent_batch})


    def calc_rule_constraints(self, rule, new_data=None, new_rule_fea=None):
        
        distr = rule.log_distribution(1,new_data,new_rule_fea)
        #print "th rule  disr_value:"+str(distr)+"\n"

        distr_all = np.maximum(np.minimum(distr, 60.), -60.) # truncate to avoid over-/under-flow
        #print "truncated and exp distr_value:"+str(np.exp(distr_all))
        return distr_all

class RobotRDQN(RobotCNNDQN):

    def __init__(self, actions=2, vocab_size=20000, max_len=120, embeddings=[]):
        super(RobotRDQN, self).__init__(actions, vocab_size, max_len, embeddings)

    def get_action(self, observation):
        #print "DQN is smart."
        self.current_state = observation[0]
        self.ind = observation[1]
        #print np.shape(self.current_state), np.shape(self.ind), self.ind
        qvalue = self.sess.run(self.qvalue, feed_dict={self.state: [ self.current_state ]})[0]

        action = np.zeros(self.action)
        action_index = 0
        # if self.timeStep % FRAME_PER_ACTION == 0:
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.action)
            action[action_index] = 1
        elif self.ind == 1:
            action[1] = 1
        else:
            action_index = np.argmax(qvalue)
            action[action_index] = 1
        # else:
        #    action[0] = 1 # do nothing

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.time_step > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action




