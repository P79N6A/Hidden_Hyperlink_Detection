#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
from bs4 import BeautifulSoup
import urllib.request
import os
import codecs
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import  datasets
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import roc_curve,roc_auc_score

np.set_printoptions(suppress=True)

#获得超链接特征和文本特征
def get_key_list(A_filename,Text_filename):
    d_text_key_list=list()
    d_href_key_list=list()
    with open(A_filename,'r') as f1:
        for i in f1:
            d_href_key_list.append(i.strip('\n'))
    with open(Text_filename,'r') as f2:
        for j in f2.readlines():
            d_text_key_list.append(j.strip('\n'))
    return d_href_key_list,d_text_key_list



#遍历文件，获得MD5目录和标签y
def traverse_directory(WebDirectory,mainfile):
    MD5_list=list()
    flag_list=list()
    with open(mainfile,'r') as f:
        for i in f:
            flag=i.split(',',4)[1]
            if flag=='d':
                flag_list.append(1)
                MD5 = i.split(',', 4)[2]
                each_file = os.path.join(WebDirectory, MD5)
                MD5_list.append(each_file)
            elif flag=='n':
                flag_list.append(0)
                MD5 = i.split(',', 4)[2]
                each_file = os.path.join(WebDirectory, MD5)
                MD5_list.append(each_file)
    return MD5_list,flag_list

def read_file(filename):
    try:
        f=codecs.open(filename,'r',encoding='utf-8')
        Web_data=f.readlines()
        # print(filename)
        # print('******')
        Web_data = '\n'.join(Web_data)
        f.close()
    except:
        f=codecs.open(filename,'r',encoding='gb18030')
        Web_data = f.readlines()
        # print(Web_data)
        # print(filename)
        # print('******')
        Web_data = '\n'.join(Web_data)
        f.close()
    return Web_data

def parse_HTML(Web_data):
    try:
        soup = BeautifulSoup(Web_data, "html.parser")
        doc_length=len(Web_data)

        #黑链结构特征
        d_fea_list = [Web_data.count(fea) for fea in D_Feature]
        d_t1=np.asarray(d_fea_list)
        d_fea_vector=np.hstack((d_t1,[d_t1.sum(), d_t1.max(), d_t1.std(), d_t1.mean(), sum(d_t1 > 0)]))

        #黑链文本特征
        d_text_list=[Web_data.count(text) for text in d_text_key_list]
        d_t2=np.asarray(d_text_list)
        d_text_vector=np.hstack((d_t2,[d_t2.sum(),d_t2.max(),d_t2.std(),d_t2.mean(),sum(d_t2>0)]))

        #黑链超链接特征
        d_href_list=[Web_data.count(href) for href in d_href_key_list]
        d_t3=np.asarray(d_href_list)
        d_href_vector=np.hstack((d_t3,[d_t3.sum(),d_t3.max(),d_t3.std(),d_t3.mean(),sum(d_t3>0)]))

        #A标签中字符数，以及所有A标签的字符数占文本总字符数的比例（10维）
        tot_a_soup=soup.findAll('a')
        a_list = [0] * 10
        total_a = len(tot_a_soup)
        if total_a != 0:
            a_len_list = [len(a_soup) for a_soup in tot_a_soup]
            a_all_per = sum(a_len_list) / float(doc_length)

            #div style中A标签占得比例=div style中超链接总数/所有链接总数
            div_style_soup = soup.findAll(lambda tag: tag.name == 'div' and 'style' in tag.attrs and len(tag.attrs) == 1)
            div_style_a_len_list = [len(a_soup.findAll(lambda tag: tag.name == 'a' and len(tag.attrs) <= 2 and ('href' in tag.attrs))) for a_soup in
            div_style_soup]
            div_style_a_num = sum(div_style_a_len_list)
            div_style_a_per = div_style_a_num / float(total_a)

            #div id中A标签占得比例=div id中超链接总数/所有链接总数
            div_id_soup = soup.findAll(lambda tag: tag.name == 'div' and 'id' in tag.attrs and len(tag.attrs) == 1)
            div_id_a_len_list = [len(a_soup.findAll(lambda tag: tag.name == 'a' and len(tag.attrs) <= 2 and ('href' in tag.attrs))) for a_soup in
            div_id_soup]
            div_id_a_num = sum(div_id_a_len_list)
            div_id_a_per = div_id_a_num / float(total_a)

            #div class中A标签占得比例=div class中超链接总数/所有链接总数
            div_class_soup = soup.findAll(lambda tag: tag.name == 'div' and 'class' in tag.attrs and len(tag.attrs) == 1)
            div_class_a_len_list = [len(a_soup.findAll(lambda tag: tag.name == 'a' and len(tag.attrs) <= 2 and ('href' in tag.attrs))) for a_soup in
            div_class_soup]
            div_class_a_num = sum(div_class_a_len_list)
            div_class_a_per = div_class_a_num / float(total_a)

            #td、marquee、ul中A标签占得比例=td、marquee、ul中超链接总数/所有链接总数
            mix_soup = soup.findAll(lambda tag: tag.name == 'td' or tag.name == 'marquee' or tag.name == 'ul')
            mix_a_len_list = [len(a_soup.findAll(lambda tag: tag.name == 'a' and len(tag.attrs) <= 2 and ('href' in tag.attrs)))
            for a_soup in mix_soup]
            mix_a_num = sum(mix_a_len_list)
            mix_a_per = mix_a_num / float(total_a)

            a_list = [total_a, a_all_per, div_style_a_num, div_style_a_per, div_id_a_num, div_id_a_per, div_class_a_num,
            div_class_a_per, mix_a_num, mix_a_per]
            d_a_np = np.asarray(a_list)
        return np.hstack((d_fea_vector,d_text_vector,d_href_vector,d_a_np ))
    except:
        # print(a_line)
        return np.asarray([0]*(21+205+206+5+10))

def test_GradientBoostingClassifier(*data):
    '''
    测试 GradientBoostingClassifier 的用法

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    clf=ensemble.GradientBoostingClassifier(n_estimators=120,max_depth=6,max_leaf_nodes=None,learning_rate=0.75,subsample=0.96)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    print("Traing Score:%f"%clf.score(X_train,y_train))
    # print("Testing Score:%f"%clf.score(X_test,y_test))
    return y_pred, y_test,y_score

def test_DecisionTreeClassifier(*data):
    '''
    测试 DecisionTreeClassifier 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:,1]

    print("Training score:%f"%(clf.score(X_train,y_train)))
    print("Testing score:%f"%(clf.score(X_test,y_test)))
    return y_test,y_score
def fig_plot(y_test_1, y_score_1,y_score_2,y_score_3):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # fpr, tpr = {}, {}
    # roc_auc = {}
    fpr1, tpr1, thresholds1 = roc_curve(y_test_1, y_score_1)
    fpr2, tpr2, thresholds2 = roc_curve(y_test_1, y_score_2)
    fpr3, tpr3, thresholds3 = roc_curve(y_test_1, y_score_3)
    auc1 = roc_auc_score(y_test_1, y_score_1)
    auc2 = roc_auc_score(y_test_1, y_score_2)
    auc3 = roc_auc_score(y_test_1, y_score_3)

    ax.plot(fpr1, tpr1, label="CART,auc=%s" % (auc1),color='green')
    ax.plot(fpr2, tpr2, label="GBDT,auc=%s" % (auc2), color='black')
    ax.plot(fpr3, tpr3, label="RF,auc=%s" % (auc3), color='red')
    ax.plot([0,1],[0,1],'k--',color='grey')


    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")
    ax.set_xlim = (0, 1)
    ax.set_ylim = (0, 1)
    plt.show()

def test_RandomForestClassifier(*data):
    '''
    测试 RandomForestClassifier 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    clf=ensemble.RandomForestClassifier(bootstrap=True,criterion='gini',max_depth=50,n_estimators=142)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    print("Traing Score:%f"%clf.score(X_train,y_train))
    # print("Testing Score:%f"%clf.score(X_test,y_test))
    return y_pred, y_test,y_score



if __name__=='__main__':
    D_Feature = [
        'padding:0px',
        'from f-l',
        'style="display:none',
        'top:',
        'left:',
        'right:',
        'fixed;',
        "style.display='none'",
        'document.getElementById',
        'color="#000000"',
        'color:#FFFFFF',
        'absolute;',
        '<marquee',
        'document.write',
        '<div class="cont-wrap" style="line-height:26px">',
        'style="text-decoration: none'
    ]
    A_filename='Unique_BlackA.txt'
    Text_filename='Unique_Blackname.txt'
    mainfile = 'data.txt'
    WebDirectory = './file/'
    MD5_list, flag_list=traverse_directory(WebDirectory,mainfile)
    d_href_key_list, d_text_key_list=get_key_list(A_filename,Text_filename)
    X=list()
    for aline in MD5_list:
        Web_data = read_file(aline)
        per_vector=parse_HTML(Web_data)
        X.append(per_vector)
        # print(per_vector)
        # print(len(per_vector))
        print('************************')
    X=np.asarray(X)
    Y=np.asarray(flag_list)
    X_train, X_test, y_train, y_test=train_test_split(X, Y, test_size=0.25,random_state=0,stratify=Y)
    y_test_1, y_score_1 = test_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
    y_pred_2, y_test_2, y_score_2 =test_GradientBoostingClassifier(X_train, X_test, y_train, y_test)
    y_pred_3, y_test_3, y_score_3 = test_RandomForestClassifier(X_train, X_test, y_train, y_test)
    fig_plot(y_test_1, y_score_1, y_score_2, y_score_3)
