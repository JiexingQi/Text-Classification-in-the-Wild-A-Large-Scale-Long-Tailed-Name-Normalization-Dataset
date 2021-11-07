from tqdm import tqdm_notebook as tqdm
import pickle

trainpath='/home/datamerge/ACL/Data/210422/train/train_part.txt'
devpath='/home/datamerge/ACL/Data/210422/dev/dev.txt'
testpath='/home/datamerge/ACL/Data/210422/test/test.txt'
newtrainpath='newtrain.txt'
newdevpath='newdev.txt'
newtestpath='newtest.txt'
label2numpath="label2num.pkl"
num2labelpath="num2label.pkl"
labels = set()
label2num = {}
num2label = {}

def writeNewTXT(path,newpath): #将label变为对应ID并且写入新txt
    f = open(newpath,'w')
    with open(path,encoding = 'utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines):
            words = line.split('\t\t',1)
            words[1] = words[1].replace('\n','')
            if(label2num.__contains__(words[1])):
                f.write(words[0])
                f.write('\t')
                f.write("__label__")
                f.write(str(label2num[words[1]]))
                f.write('\n')
    f.close()

with open(train_path,encoding = 'utf-8') as file:
    lines = file.readlines()
    for line in tqdm(lines):
        words = line.split('\t\t',1)
        words[1] = words[1].replace('\n','')
        labels.add(words[1])     #读入标签和特征
labels = list(labels)

for i in range(len(labels)): 
    label2num[labels[i]] = i
    num2label[i] = labels[i]
    
pickle.dump(label2num,open(label2numpath,'wb'))
pickle.dump(num2label,open(num2labelpath,'wb'))  #做标签数字映射并写入文件

writeNewTXT(train_path,newtrainpath)
writeNewTXT(devpath,newdevpath)
writeNewTXT(testpath,newtestpath)