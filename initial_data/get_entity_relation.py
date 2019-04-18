import os 
import sys
import collections


def get_entity_embedding():
    entity_vec=[]
    with open ("../data/entity2vec.bern") as f:
        for line in f:
            entity_vec.append(map(lambda x:float(x) ,line.strip().split("\t")))
    return entity_vec

def get_entity2id():
    entity2id = {}
    with open ("../data/entity2id.txt") as f:
        for line in f:
            en_id=line.strip().split()
            entity2id[en_id[0]]=int(en_id[1])
    return entity2id

def get_en(en):
    en_1=list(en)
    en_1[1]='/'
    en_2=''.join(en_1)
    return '/'+en_2

en_vec=get_entity_embedding()
en2id=get_entity2id()

def get_entityre(en1,en2):
    re=[]
    for i in range(len(en_vec[en2id[en1]])):
        re.append(en_vec[en2id[en1]][i]-en_vec[en2id[en2]][i])
    return re   
 
if __name__ == "__main__":
    a=get_entityre("m.0ccvx","m.05gf08")
    print a

