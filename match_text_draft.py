#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
from sklearn.feature_selection import SelectKBest,f_classif
from bs4 import BeautifulSoup
import urllib.request
import os
import codecs
from sklearn.tree import DecisionTreeClassifier
from sklearn import  datasets
from sklearn.model_selection import train_test_split



def traverse_directory(WebDirectory,mainfile):
    MD5_list=list()
    flag_list=list()
    with open(mainfile,'r') as f:
        for i in f:
            flag=i.split(',',4)[1]
            MD5=i.split(',',4)[2]
            each_file=os.path.join(WebDirectory,MD5)
            MD5_list.append(each_file)
            if flag=='d':
                flag_list.append(1)
            else:
                flag_list.append(0)
    return MD5_list,flag_list

def read_file(filename):
    try:
        f=codecs.open(filename,'r',encoding='utf-8')
        Web_data=f.readlines()
        # print(filename)
        # print('******')
        f.close()
        Web_data = '\n'.join(Web_data)
    except:
        f=codecs.open(filename,'r',encoding='gb18030')
        Web_data = f.readlines()
        # print(Web_data)
        # print(filename)
        # print('******')
        f.close()
        Web_data = '\n'.join(Web_data)
    return Web_data

if __name__=='__main__':
    try:
        f8=open('text_index.txt','w')
        f9=open('text_vector.txt','w')
        X=list()
        d_href_key_list = list()
        with open('Blackname.txt', 'r') as f1:
            for i in f1:
                d_href_key_list.append(i.strip('\n'))
        mainfile = 'file_list_20170430_new.txt'
        WebDirectory = './file/'
        MD5_list, flag_list = traverse_directory(WebDirectory, mainfile)
        for aline in MD5_list:
            Web_data = read_file(aline)
            A_feature_list=[Web_data.count(fea) for fea in d_href_key_list]
            X.append(A_feature_list)
            print('**')
        selector=SelectKBest(score_func=f_classif,k=200)
        selector.fit(X,flag_list)
        print(selector.get_support(True))
        print(selector.transform(X))
        f8.write(selector.get_support(True))
        f9.write(selector.transform(X))
        f8.close()
        f9.close()
    except:
        print(aline)