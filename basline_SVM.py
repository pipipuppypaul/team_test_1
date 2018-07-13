##################################
######## baseline 2

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
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
    test_pred = pd.DataFrame()
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
#             clf = LogisticRegression(C=4, dual=True)
            clf = svm.LinearSVC() ##不过是换了个模型而已，我觉得没有改变本质
            clf.fit(trn_term_doc, y)
#             preds=clf.predict_proba(test_term_doc)
            preds=clf.predict(test_term_doc)
            preds = pd.DataFrame(preds,columns=['class'])
            preds["id"]=list(test_id["id"])
            preds["class"]=(preds["class"]+1).astype(int)
#             print(preds)

            #生成提交结果
#             test_pred = pd.concat([test_pred,preds],ignore_index=True)
            
#             test_pred["class"]=(test_pred["class"]+1).astype(int)
#             print(test_pred.shape)
#             print(test_id.shape)
            
            if head ==0:
                preds[["id","class"]].to_csv('../article_classification/sub_SVM_baseline.csv',mode='a',index=None)
                head = 1
            elif head ==1:
                preds[["id","class"]].to_csv('../article_classification/sub_SVM_baseline.csv',mode='a',index=None,header=False)
            t2=time.time()
            print('batch No.',i-1,", time use:",t2-t1)
        except StopIteration: 
            loop = False 
            print("Iteration is stopped.")
#     print(test_pred)
    
    
process(size=6000)