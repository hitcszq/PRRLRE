import sys
import argparse
from dqnrobot import RobotCNNDQN,RobotPRDQN,RobotRDQN
from pg_reinforce import PolicyGradientREINFORCE,PRPolicyGradientREINFORCE
import numpy as np
import tensorflow as tf
import random
from  environment import environment ,environment_ctl
from generate_new_data import generate_new_test ,generate_new_train
from train_GRU import train_GRU
from test_GRU import test_GRU
from collections import deque

tf.flags.DEFINE_integer("max_seq_len", 120, "sequence")
tf.flags.DEFINE_integer("max_vocab_size", 20000, "vocabulary")

#FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
#print("\nParameters:")
#for attr, value in sorted(FLAGS.__flags.items()):
#    print("{}={}".format(attr.upper(), value))
#print("")

AGENT = "PRPGRL"
MAX_EPISODE = 210
robot_data="train_robot_"
base_model_train = "train_"
base_model_test = "testall_"
state_dim = 330
iterative_num=2
def policy_network(states):
  # define policy neural network
    W1 = tf.get_variable("W1", [state_dim, 20],
                       initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [20],
                       initializer=tf.constant_initializer(0))
    h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
    W2 = tf.get_variable("W2", [20, 2],
                       initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", [2],
                       initializer=tf.constant_initializer(0))
    p = tf.matmul(h1, W2) + b2
    return p

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help="require a decision agent")
    parser.add_argument(
        '--episode', help="require a maximum number of playing the game")
    parser.add_argument('--train_robot', help="training phase")
    parser.add_argument('--train_base_model', help="testing phase")
    parser.add_argument('--test_base_model', help="testing phase")

    args = parser.parse_args()
    global AGENT, MAX_EPISODE
    AGENT = args.agent
    MAX_EPISODE = int(args.episode)
    robot_data = args.train_robot
    base_model_train = args.train_base_model
    base_model_train = args.train_base_model


def initialise_game(mode,sf):
    # load game
    print("Loading game ..")
    env_ctl=environment_ctl("./data/%sword.npy"%mode,"./data/%spos1.npy"%mode,"./data/%spos2.npy"%mode,\
         "./data/%sind.npy"%mode, "./data/%sentity.npy"%mode , "./data/%sy.npy"%mode,sf)
    
    return env_ctl

def play_ner():
    actions = 2
    global AGENT

    global robot_data
    env_ctl = initialise_game(robot_data,True)
        # initialise a decision robot
    episode = 1
    sess = tf.Session()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    episode_history = deque(maxlen=100)
    total_rewards=0
 
    mean_rw=[]
    print(">>>>>> Playing game ..")
    if AGENT == "random":
        robot = RobotRandom(actions)
    elif AGENT == "DQN":
        env = env_ctl.get_new_environment()
        observation=env.first_observation()
        robot = RobotCNNDQN(actions)
        while episode <= MAX_EPISODE:
            #print '>>>>>>> Current game round ', episode, 'Maximum ', MAX_EPISODE
            action = robot.get_action(observation)
            #print '> Action', action
            observation2, reward, terminal, info = env.step(action)

            total_rewards += reward
            #print '> Reward', reward
            robot.update(observation, action, reward, observation2, terminal)
            observation = observation2
            if terminal == True:
                env=env_ctl.get_new_environment()
                observation=env.first_observation()
                episode_history.append(total_rewards)
                mean_rewards = np.mean(episode_history)
                print mean_rewards
                mean_rw.append(mean_rewards)
                total_rewards = 0
                episode += 1
                print '> Terminal <'
    #mean_rw=np.array(mean_rw)
    #np.save('./data/mean_rw.npy',mean_rw)
    elif AGENT == "RDQN":
        env = env_ctl.get_new_PR_environment()

        observation=env.first_observation()
        robot = RobotRDQN(actions)
        while episode <= MAX_EPISODE:
            #print '>>>>>>> Current game round ', episode, 'Maximum ', MAX_EPISODE
            action = robot.get_action(observation)
            print '> Action', action
            observation2, reward, terminal, info= env.step(action)
            print '> Reward', reward
            robot.update(observation[0], action, reward, observation2[0], terminal)
            observation = observation2
            if terminal == True:
                env=env_ctl.get_new_PR_environment()
                observation=env.first_observation()
                episode += 1
                print '> Terminal <'
    elif AGENT == "PGRL":
        env = env_ctl.get_new_environment()
        observation=np.reshape(env.first_observation(),(1,state_dim))
        robot = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       actions,
                                       summary_writer= None )
        saver = tf.train.Saver(max_to_keep=None)
        while episode <= MAX_EPISODE:
            print '>>>>>>> Current game round ', episode, 'Maximum ', MAX_EPISODE
            action = robot.sampleAction(observation)
            print '> Action', action
            observation2, reward, terminal, info = env.step(action)
            print '> Reward', reward
            robot.storeRollout(observation, action, reward)
            observation = np.reshape(observation2,(1,state_dim))
            if terminal == True:
                robot.updateModel()
                env=env_ctl.get_new_environment()
                if env == None:
                    env_ctl = initialise_game(robot_data,True)
                    env=env_ctl.get_new_PR_environment()
                observation=np.reshape(env.first_observation(),(1,state_dim))
                episode += 1
                if episode % 1000 == 0:
                    print "saving model"+"\n"
                    path = saver.save(sess, "./model/%s_robot_model"%AGENT, global_step=episode)
                    tempstr = 'have saved model to '+path
                    print tempstr
                print '> Terminal <'

    elif AGENT == "PRPGRL":
        env = env_ctl.get_new_PR_environment()

        observation=(np.reshape(env.first_observation()[0],(1,state_dim)),env.first_observation()[1])
        robot = PRPolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       actions,
                                       summary_writer= None)
        saver = tf.train.Saver(max_to_keep=None)
        while episode <= MAX_EPISODE:
            print '>>>>>>> Current game round ', episode, 'Maximum ', MAX_EPISODE
            action = robot.sampleAction(observation)
            print '> Action', action
            observation2, reward, terminal, info = env.step(action)
            print '> Reward', reward
            robot.storeRollout(observation[0], action, reward)
            observation = (np.reshape(observation2[0],(1,state_dim)),observation2[1])
            if terminal == True:
                robot.updateModel()
                env=env_ctl.get_new_PR_environment()
                if env == None:
                    env_ctl = initialise_game(robot_data,True)
                    env=env_ctl.get_new_PR_environment()    
                observation=(np.reshape(env.first_observation()[0],(1,state_dim)),env.first_observation()[1])
                episode += 1
                if episode % 1000 == 0:
                    print "saving model"+"\n"
                    path = saver.save(sess,'./model/%s_robot_model'%AGENT,global_step=episode)
                    tempstr = 'have saved model to '+path
                    print tempstr
                print '> Terminal <'
    else:
        print "** There is no robot."
        raise SystemExit
    return robot

def test_agent_batch(robot, game):
    select_action = []
    episode_action=[]
    episode_number = 0
    global AGENT 
    if AGENT == "DQN":
        env = game.get_new_environment()
        observation=env.first_observation()
        while True:
            action = robot.get_action(observation)
            episode_action.append(action)       
            observation, reward, terminal, info = env.step(action)
            if terminal:
                episode_number += 1
                env = game.get_new_environment()
                if env==None:
                    select_action.append(episode_action)
                    break
                else:
                    select_action.append(episode_action)
                    print episode_action
                    episode_action = []
                    observation=env.first_observation() 
    elif AGENT == "RDQN":
        env = game.get_new_PR_environment()
        observation=env.first_observation()
        while True:
            action = robot.get_action(observation)
            episode_action.append(action)
            observation, reward, terminal, info = env.step(action)
            if terminal:
                episode_number += 1
                env = game.get_new_PR_environment()
                if env==None:
                    select_action.append(episode_action)
                    break
                else:
                    select_action.append(episode_action)
                    print episode_action
                    episode_action = []
                    observation=env.first_observation()
    elif AGENT == "PGRL":
        env = game.get_new_environment()
        observation=np.reshape(env.first_observation(),(1,state_dim))
        while True:
            action = robot.sampleActiontest(observation)
            episode_action.append(action)
            observation, reward, terminal, info = env.step(action)
            observation = np.reshape(observation ,(1,state_dim))
            if terminal:
                episode_number += 1
                env = game.get_new_environment()
                if env==None:
                    select_action.append(episode_action)
                    break
                else:
                    select_action.append(episode_action)
                    print episode_action
                    episode_action = []
                    observation=np.reshape(env.first_observation(),(1,state_dim))
    elif AGENT == "PRPGRL":
        env = game.get_new_PR_environment()
        observation = (np.reshape(env.first_observation()[0],(1,state_dim)),env.first_observation()[1])
        while True:
            action = robot.sampleActiontest(observation)
            episode_action.append(action)
            observation, reward, terminal, info = env.step(action)
            observation = (np.reshape( observation[0],(1,state_dim)),observation[1])
            if terminal:
                episode_number += 1
                env = game.get_new_PR_environment()
                if env==None:
                    select_action.append(episode_action)
                    break
                else:
                    select_action.append(episode_action)
                    #print episode_action
                    episode_action = []
                    observation=env.first_observation()
                    observation = (np.reshape(env.first_observation()[0],(1,state_dim)),env.first_observation()[1])
    return select_action

def test():
    global AGENT
    global base_model_train,base_model_test
    robot = play_ner() 
    test_train_env = initialise_game(base_model_train,False)
    #test_test_env = initialise_game(base_model_test)
    select_train = test_agent_batch(robot, test_train_env)
    #select_test = test_agent_batch(robot, test_test_env)
    select_train = np.array(select_train)
    #select_test = np.array(select_test)
    np.save('./data/select_train.npy',select_train)
    #np.save('./data/select_test.npy',select_test)
    select_train = np.load('./data/select_train.npy')
    #select_test=np.load('./data/select_test.npy')
    generate_new_train(select_train) # train triple
    #generate_new_test(select_test)
    train_GRU(AGENT)
    test_GRU(AGENT)

def test_interative():
    global AGENT
    global base_model_train,iterative_num
    for iter_i in range(iterative_num):
        robot = play_ner() 
        test_train_env = initialise_game(base_model_train,False)
        print str(iter_i)+" iteration>>>>>>>>>>>>>>>>>>"
        select_train = test_agent_batch(robot, test_train_env)
        select_train = np.array(select_train)
        np.save('./data/select_train.npy',select_train)
        select_train = np.load('./data/select_train.npy')
        generate_new_train(select_train) # train triple
        train_GRU(AGENT)
    
    test_GRU(AGENT)

def main():
    #parse_args()
    # play games for training a robot
    #robot = play_ner()
    # play a new game with the trained robot
    test_interative()

if __name__ == '__main__':
    main()
