import fasttext
from tqdm import tqdm_notebook as tqdm
import pickle
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
e_num = 20
lr_num = 0.5
trainpath='newtrain.txt'
devpath='newdev.txt'
testpath='newtest.txt'
thread_num = 30
nor2len_path = "/home/datamerge/ACL/Data/210422/pkl/210422_nor2len_dict.pkl"
num2nor_path =  "num2label.pkl"

model = fasttext.train_supervised(input=trainpath,thread = thread_num,epoch=e_num,lr=lr_num) ##训练模型
model.save_model("fasttext{}epoch{}lr.bin".format(e_num,lr_num))
model = fasttext.load_model("fasttext{}epoch{}lr.bin".format(e_num,lr_num))
nor2len = pickle.load(open(nor2len_path, 'rb'))
num2nor = pickle.load(open(num2nor_path, 'rb'))

def get_diff(h2m, m2l, real_result,predict_result):
    real_h= []
    real_m= []
    real_l= []
    predict_h = []
    predict_m = []
    predict_l = []
    for i in range(len(real_result)):
        t = num2nor[real_result[i]]
        if nor2len[t] < m2l:
            real_l.append(real_result[i])
            predict_l.append(predict_result[i])
        elif nor2len[t] <= h2m:
            real_m.append(real_result[i])
            predict_m.append(predict_result[i])
        else:
            real_h.append(real_result[i])
            predict_h.append(predict_result[i])
    return real_h,real_m,real_l,predict_h,predict_m,predict_l


def getscore(path):
    r_label=[]
    predict=[]
    with open(path,encoding = 'utf-8') as file:
        lines = file.readlines()
        for line in tqdm(lines):
            words = line.split('\t',1)
            result = model.predict(words[0])
            predict.append(int(result[0][0][9:]))
            words[1] = words[1].replace('\n','').replace('__label__','')
            r_label.append(int(words[1]))
    label_h,label_m,label_l,predict_h,predict_m,predict_l = get_diff(20,5,r_label,predict)
    return [accuracy_score(r_label, predict),accuracy_score(label_h, predict_h),accuracy_score(label_m, predict_m),accuracy_score(label_l, predict_l)],[precision_score(r_label, predict,average='macro'),precision_score(label_h, predict_h,average='macro'),precision_score(label_m, predict_m,average='macro'),precision_score(label_l, predict_l,average='macro')],[recall_score(r_label, predict,average='macro'),recall_score(label_h, predict_h,average='macro'),recall_score(label_m, predict_m,average='macro'),recall_score(label_l, predict_l,average='macro')],[f1_score(r_label, predict,average='macro'),f1_score(label_h, predict_h,average='macro'),f1_score(label_m, predict_m,average='macro'),f1_score(label_l, predict_l,average='macro')]



a1,p1,r1,f1 = getscore(devpath)
a2,p2,r2,f2 = getscore(testpath)


with open('fasttext{}epoch{}lr.txt'.format(e_num,lr_num),'w') as fi:
    fi.write('valid:\n')
    for a,p,r,f in zip(a1,p1,r1,f1):
        fi.write('accuary:'+str(a)+'\n')
        fi.write('precision:'+str(p)+'\n')
        fi.write('recall:'+str(r)+'\n')
        fi.write('f1:'+str(f)+'\n\n')

    fi.write('test:\n')
    for a,p,r,f in zip(a2,p2,r2,f2):
        fi.write('accuary:'+str(a)+'\n')
        fi.write('precision:'+str(p)+'\n')
        fi.write('recall:'+str(r)+'\n')
        fi.write('f1:'+str(f)+'\n\n')


# In[ ]:




