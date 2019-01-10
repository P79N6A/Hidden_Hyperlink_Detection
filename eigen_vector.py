#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
from bs4 import BeautifulSoup
import time
import urllib.request
import os
import codecs
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import  datasets
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import matplotlib.pyplot as plt

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


def test_DecisionTreeClassifier(*data):
    '''
    测试 DecisionTreeClassifier 的用法

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    middle = time.clock()
    print(middle-start)

    clf.predict(X_test)
    print("Training score:%f"%(clf.score(X_train,y_train)))
    print("Testing score:%f"%(clf.score(X_test,y_test)))
    end = time.clock()
    print(end-middle)

def test_DecisionTreeClassifier_criterion(*data):
    '''
    测试 DecisionTreeClassifier 的预测性能随 criterion 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    criterions=['gini','entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train, y_train)
        print("criterion:%s"%criterion)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_splitter(*data):
    '''
    测试 DecisionTreeClassifier 的预测性能随划分类型的影响

    :param data: 可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    splitters=['best','random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train, y_train)
        print("splitter:%s"%splitter)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))
def test_DecisionTreeClassifier_depth(*data,maxdepth):
    '''
    测试 DecisionTreeClassifier 的预测性能随 max_depth 参数的影响

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :param maxdepth: 一个整数，用于 DecisionTreeClassifier 的 max_depth 参数
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    depths=np.arange(1,maxdepth)
    training_scores=[]
    testing_scores=[]
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))

    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label="Training score",color='r', linestyle='--')
    ax.plot(depths,testing_scores,label="Testing score",color='b')
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5,loc='best')
    plt.grid(axis='y')
    plt.show()















if __name__=='__main__':
    start = time.clock()
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
    test_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
    # test_DecisionTreeClassifier_criterion(X_train, X_test, y_train, y_test)  # 调用 test_DecisionTreeClassifier_criterion
    # test_DecisionTreeClassifier_splitter(X_train, X_test, y_train, y_test)  # 调用 test_DecisionTreeClassifier_splitter
    test_DecisionTreeClassifier_depth(X_train, X_test, y_train, y_test,maxdepth=30)  # 调用 test_DecisionTreeClassifier_depth
