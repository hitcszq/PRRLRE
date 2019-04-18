import os
import sys
def generate_reid():
    re_id=[]
    with open ("./relation2id.txt","r") as f:
        for line in f:
            re_id.append(line.strip().split())

    with open ("./relation2id.txtE","w") as f:
        for i in re_id:
            f.write("\t".join(i)+"\n")

def generate_enid():
    i=0
    en_dict={}
    with open ("./train.txt","r") as f:
        for line in f:
            train_line=line.strip().split()
            en1=train_line[0]
            en2=train_line[1]
            if en1 in en_dict:
                pass
            else:
                en_dict[en1]=i
                i=i+1
            if en2 in en_dict:
                pass
            else:
                en_dict[en2]=i
                i=i+1

    with open ("./test.txt","r") as f:
        for line in f:
            train_line=line.strip().split()
            en1=train_line[0]
            en2=train_line[1]
            if en1 in en_dict:
                pass
            else:
                en_dict[en1]=i
                i=i+1
            if en2 in en_dict:
                pass
            else:
                en_dict[en2]=i
                i=i+1

    with open ("./entity.txtE","w") as f:
        for en,id_ in en_dict.items():
            f.write("\t".join([en,str(id_)])+"\n")

def generate_train():
    train_d=[]
    with open ("./train.txt","r") as f:
        for line in f:
            train_line=line.strip().split()
            train_d.append("\t".join([train_line[0],train_line[1],train_line[4]]))

    with open ("./test.txt","r") as f:
        for line in f:
            train_line=line.strip().split()
            train_d.append("\t".join([train_line[0],train_line[1],train_line[4]]))

    with open ("./train.txtE","w") as f:
        for i in train_d:
            f.write(i+"\n")


if __name__ == "__main__":
    generate_reid()
    generate_enid()
    generate_train()
