import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time

# 102500 * 19

def process(size,batch=3):
    head = 0
    loop = True 
    train_reader = pd.read_csv('/Volumes/KNIGHT/article_class/new_data/train_set.csv',sep=',',iterator=True)
    test_reader = pd.read_csv('/Volumes/KNIGHT/article_class/new_data/test_set.csv',sep=',',iterator=True)
    test_id_reader = pd.read_csv('/Volumes/KNIGHT/article_class/new_data/test_set.csv',sep=',',iterator=True)
    chunkSize = size   # 这个chunk大小可以改 
    i = 1
    while loop:
        i+=1
        try:
            t1=time.time()
            train = train_reader.get_chunk(chunkSize)
            test = test_reader.get_chunk(chunkSize)
            test_id = test_id_reader.get_chunk(chunkSize)[["id"]].copy()
        
            column="word_seg"
            n = train.shape[0]
            vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
            trn_term_doc = vec.fit_transform(train[column])
            test_term_doc = vec.transform(test[column])

            y=(train["class"]-1).astype(int)
            clf = LogisticRegression(C=4, dual=True)
            clf.fit(trn_term_doc, y)
            preds=clf.predict_proba(test_term_doc)

            #保存概率文件
            test_prob=pd.DataFrame(preds)
#             test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
#             test_prob["id"]=list(test_id["id"])
#             test_prob.to_csv('../article_classification/prob_lr_baseline.csv',mode='a',index=None)

            #生成提交结果
            preds=np.argmax(preds,axis=1)
            test_pred=pd.DataFrame(preds)
            test_pred.columns=["class"]
            test_pred["class"]=(test_pred["class"]+1).astype(int)
#             print(test_pred.shape)
#             print(test_id.shape)
            test_pred["id"]=list(test_id["id"])
            if head ==0:
                test_pred[["id","class"]].to_csv('../article_classification/sub_lr_baseline.csv',mode='a',index=None)
                head = 1
            elif head ==1:
                test_pred[["id","class"]].to_csv('../article_classification/sub_lr_baseline.csv',mode='a',index=None,header=False)
            t2=time.time()
            print('batch No.',i-1,", time use:",t2-t1)
        except StopIteration: 
            loop = False 
            print("Iteration is stopped.")
            
process(size=5000)