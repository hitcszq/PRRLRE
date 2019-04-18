import numpy as np
import os
import json
import re
from get_entity_relation import get_entityre
#embedding the position 
def pos_embed(x):
    if x < -60:
        return 0
    if x >= -60 and x <= 60:
        return x+61
    if x > 60:
        return 122
#find the index of x in y, if x not in y, return -1
def find_index(x,y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


def match_pattern(pattern_dict,sentence,relation):
    if relation in pattern_dict:
        pattern_list = pattern_dict[relation]
    else:
        return 0
    for pattern in pattern_list:
        pattern_string = ' '.join(pattern)
        re_string = pattern_string.replace("<JJ>",".*")
        if re.search(re_string,sentence) == None:
            continue
        else:
            return 1
    return 0

#reading data
def init():
    print 'reading pattern list...'
    pattern_dict = {}
    with open('../origin_data/nlf.json') as f:
        lfs = f.readlines()
        lfs = map(lambda t: json.loads(t), filter(lambda x: x and not x.isspace(), lfs))
        for pattern in lfs:
            words = filter(lambda t: t and not t.isspace(), pattern['Texture'].split(' '))
            re_type = filter(lambda t: t and not t.isspace(), pattern['relationType'])
            if re_type in pattern_dict:
                pattern_dict[re_type].append(words)
            else:
                pattern_dict[re_type]=[]
                pattern_dict[re_type].append(words)
    print pattern_dict
    print 'reading word embedding data...'
    vec = []
    word2id = {}
    f = open('../origin_data/vec.txt')
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    
    dim = 50
    vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
    vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
    vec = np.array(vec,dtype=np.float32)

    
    print 'reading relation to id'
    relation2id = {}    
    f = open('../origin_data/relation2id.txt','r')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    #length of sentence is 70
    fixlen = 70
    #max length of position embedding is 60 (-60~+60)
    maxlen = 60

    train_sen = {} #{entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...]}
    train_ans = {} #{entity pair:[label1,label2,...]} the label is one-hot vector
    train_tup = {}

    print 'reading train data...'
    f = open('../origin_data/train.txt','r')

    while True:
        content = f.readline()
        if content == '':
            break
        
        content = content.strip().split()
        #get entity name
        en1_c=content[0]
        en2_c=content[1]
        en1 = content[2] 
        en2 = content[3]

        re_rep=get_entityre(en1_c,en2_c)
        relation = 0
        if content[4] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[4]]
        #put the same entity pair sentences into a dict
        tup = (en1,en2)
        label_tag = 0
        if tup not in train_sen:
            train_sen[tup]=[]
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            train_tup[tup] = []
            train_tup[tup].append(re_rep)
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            
            temp = find_index(label,train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup])-1
                train_sen[tup].append([])
            else:
                label_tag = temp

        sentence = content[5:-1]
        
        en1pos = 0
        en2pos = 0
        match=match_pattern(pattern_dict, " ".join(sentence) ,content[4])
        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word,rel_e1,rel_e2,match])

        for i in range(min(fixlen,len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]
            
            output[i][0] = word

        train_sen[tup][label_tag].append(output)

    print('reading test data ...')

    match_num=0
    test_sen = {} #{entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {} #{entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)
    test_tup = {}
    f = open('../origin_data/test.txt','r')

    while True:
        content = f.readline()
        if content == '':
            break
        
        content = content.strip().split()
        en1_c=content[0]
        en2_c=content[1]
        en1 = content[2]

        re_rep=get_entityre(en1_c,en2_c)
        en2 = content[3]
        relation = 0
        if content[4] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[4]]        
        tup = (en1,en2)
        
        if tup not in test_sen:
            test_tup[tup] = re_rep
            test_sen[tup]=[]
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            test_ans[tup] = label
        else:
            y_id = relation
            test_ans[tup][y_id] = 1
            
        sentence = content[5:-1]

        en1pos = 0
        en2pos = 0
        match=match_pattern(pattern_dict, " ".join(sentence) ,content[4])
        if match == 1:
            match_num = match_num + 1 
        for i in range(len(sentence)):
            if sentence[i] == en1:
                en1pos = i
            if sentence[i] == en2:
                en2pos = i
        output = []

        for i in range(fixlen):
            word = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word,rel_e1,rel_e2,match])

        for i in range(min(fixlen,len(sentence))):
            word = 0
            if sentence[i] not in word2id:
                word = word2id['UNK']
            else:
                word = word2id[sentence[i]]

            output[i][0] = word
        test_sen[tup].append(output)
    
    train_x = []
    train_y = []
    train_en = []
    test_x = []
    test_y = []
    test_en = []

    print 'organizing train data'
    f = open('../data/train_q&a.txt','w')
    temp = 0
    for i in train_sen:
        if len(train_ans[i]) != len(train_sen[i]):
            print 'ERROR'
        lenth = len(train_ans[i])
        for j in range(lenth):
            train_en.append(train_tup[i])
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+str(np.argmax(train_ans[i][j]))+'\n')
            temp+=1
    f.close()

    print 'organizing test data'
    f = open('../data/test_q&a.txt','w')
    temp=0
    for i in test_sen:        
        #if len(test_sen[i])<3:
        #    continue
        test_en.append(test_tup[i])
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j]!=0:
                tempstr = tempstr+str(j)+'\t'
        f.write(str(temp)+'\t'+i[0]+'\t'+i[1]+'\t'+tempstr+'\n')
        temp+=1
    f.close()

    print match_num

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    train_en = np.array(train_en)
    test_en=np.array(test_en)

    np.save('../data/vec.npy',vec)
    np.save('../data/train_x.npy',train_x)
    np.save('../data/train_y.npy',train_y)
    np.save('../data/testall_x.npy',test_x)
    np.save('../data/testall_y.npy',test_y)
    np.save('../data/train_en.npy',train_en)
    np.save('../data/testall_entity.npy',test_en)

def seperate():
    
    print 'reading training data'
    x_train = np.load('../data/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []
    train_ind = []
    print 'seprating train data'
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        ind = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            temp_ind = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
                temp_ind.append(k[3])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
            ind.append(temp_ind)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)
        train_ind.append(ind)
    train_word = np.array(train_word)
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    train_ind = np.array(train_ind)
    np.save('../data/train_word.npy',train_word)
    np.save('../data/train_pos1.npy',train_pos1)
    np.save('../data/train_pos2.npy',train_pos2)
    np.save('../data/train_ind.npy',train_ind)

    print 'seperating test all data'
    x_test = np.load('../data/testall_x.npy')

    test_word = []
    test_pos1 = []
    test_pos2 = []
    test_ind = []
    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        ind = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            temp_ind = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
                temp_ind.append(k[3])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
            ind.append(temp_ind)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)
        test_ind.append(ind)
    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)
    test_ind = np.array(test_ind)

    np.save('../data/testall_word.npy',test_word)
    np.save('../data/testall_pos1.npy',test_pos1)
    np.save('../data/testall_pos2.npy',test_pos2)
    np.save('../data/testall_ind.npy',test_ind)

def getsmall():

    print 'reading training data'
    word = np.load('../data/train_word.npy')
    pos1 = np.load('../data/train_pos1.npy')
    pos2 = np.load('../data/train_pos2.npy')
    ind = np.load('../data/train_ind.npy')
    y = np.load('../data/train_y.npy')
    re = np.load('../data/train_en.npy')

    print np.shape(word),np.shape(pos1),np.shape(pos2),np.shape(ind),np.shape(y),np.shape(re)
    new_word = []
    new_pos1 = []
    new_pos2 = []
    new_y = []
    new_ind = []
    new_re = []

    #we slice some big batch in train data into small batches in case of running out of memory
    print 'get small training data'
    for i in range(len(word)):
        lenth = len(word[i])
        if lenth <= 1000:

            new_word.append(word[i])
            new_pos1.append(pos1[i])
            new_pos2.append(pos2[i])
            new_y.append(y[i])
            new_ind.append(ind[i])
            new_re.append(re[i])

        if lenth > 1000 and lenth < 2000:

            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:])

            new_y.append(y[i])
            new_y.append(y[i])

            new_ind.append(ind[i][:1000])
            new_ind.append(ind[i][1000:])

            new_re.append(re[i])
            new_re.append(re[i])

        if lenth > 2000 and lenth < 3000:
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])

            new_ind.append(ind[i][:1000])
            new_ind.append(ind[i][1000:2000])
            new_ind.append(ind[i][2000:])

            new_re.append(re[i])
            new_re.append(re[i])
            new_re.append(re[i])

        if lenth > 3000 and lenth < 4000:
            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:3000])
            new_word.append(word[i][3000:])

            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:3000])
            new_pos1.append(pos1[i][3000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:3000])
            new_pos2.append(pos2[i][3000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])

            new_ind.append(ind[i][:1000])
            new_ind.append(ind[i][1000:2000])
            new_ind.append(ind[i][2000:3000])
            new_ind.append(ind[i][3000:])

            new_re.append(re[i])
            new_re.append(re[i])
            new_re.append(re[i])
            new_re.append(re[i])

        if lenth > 4000:

            new_word.append(word[i][:1000])
            new_word.append(word[i][1000:2000])
            new_word.append(word[i][2000:3000])
            new_word.append(word[i][3000:4000])
            new_word.append(word[i][4000:])
            
            new_pos1.append(pos1[i][:1000])
            new_pos1.append(pos1[i][1000:2000])
            new_pos1.append(pos1[i][2000:3000])
            new_pos1.append(pos1[i][3000:4000])
            new_pos1.append(pos1[i][4000:])

            new_pos2.append(pos2[i][:1000])
            new_pos2.append(pos2[i][1000:2000])
            new_pos2.append(pos2[i][2000:3000])
            new_pos2.append(pos2[i][3000:4000])
            new_pos2.append(pos2[i][4000:])

            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])
            new_y.append(y[i])

            new_ind.append(ind[i][:1000])
            new_ind.append(ind[i][1000:2000])
            new_ind.append(ind[i][2000:3000])
            new_ind.append(ind[i][3000:4000])
            new_ind.append(ind[i][4000])

            new_re.append(re[i])
            new_re.append(re[i])
            new_re.append(re[i])
            new_re.append(re[i])
            new_re.append(re[i])

    new_word = np.array(new_word)
    new_pos1 = np.array(new_pos1)
    new_pos2 = np.array(new_pos2)
    new_y = np.array(new_y)
    new_ind=np.array(new_ind)

    np.save('../data/train_word.npy',new_word)
    np.save('../data/train_pos1.npy',new_pos1)
    np.save('../data/train_pos2.npy',new_pos2)
    np.save('../data/train_y.npy',new_y)
    np.save('../data/train_ind.npy',new_ind)
    np.save('../data/train_entity.npy',new_re)

                                           
#get answer metric for PR curve evaluation
def getans():
    test_y = np.load('../data/testall_y.npy')
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y,(-1))
    np.save('../data/allans.npy',allans)

def get_metadata():
    fwrite = open('../data/metadata.tsv','w')
    f = open('../origin_data/vec.txt')
    f.readline()
    while True:
        content = f.readline().strip()
        if content == '':
            break
        name = content.split()[0]
        fwrite.write(name+'\n')
    f.close()
    fwrite.close()


init()
seperate()
getsmall()
getans()
get_metadata()

