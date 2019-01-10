#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from bs4 import BeautifulSoup
import urllib.request
import os
import codecs
import numpy as np
from sklearn import ensemble
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
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

def test_GradientBoostingClassifier(*data):
    '''
    测试 GradientBoostingClassifier 的用法

    :param data:  可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    clf=ensemble.GradientBoostingClassifier(n_estimators=120,max_depth=6,max_leaf_nodes=None,learning_rate=0.75,subsample=0.96)
    clf.fit(X_train,y_train)
    middle = time.clock()
    print(middle-start)
    y_pred = clf.predict(X_test)
    print("Traing Score:%f"%clf.score(X_train,y_train))
    # print("Testing Score:%f"%clf.score(X_test,y_test))
    end = time.clock()
    print(end-middle)
    return y_pred, y_test
def test_GradientBoostingClassifier_num(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随 n_estimators 参数的影响

    :param data:   可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return: None
    '''
    X_train,X_test,y_train,y_test=data
    nums=np.arange(1,150,step=3)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for num in nums:
        clf=ensemble.GradientBoostingClassifier(n_estimators=num)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    print("training score_num:", training_scores)
    print("testing score_num:", testing_scores)
    ax.plot(nums,training_scores,label="Training Score", color='r', linestyle='--')
    ax.plot(nums,testing_scores,label="Testing Score",color='b')
    ax.set_xlabel("Estimator Number")
    ax.set_ylabel("Accuracy Score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4,1.05)
    plt.grid(axis='y')
    plt.suptitle("Gradient Boosting Classifier")
    plt.savefig('figure1.jpg')
def test_GradientBoostingClassifier_maxdepth(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随 max_depth 参数的影响

    :param data:    可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    maxdepths=np.arange(1,30)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for maxdepth in maxdepths:
        clf=ensemble.GradientBoostingClassifier(max_depth=maxdepth,max_leaf_nodes=None)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    print("training score_maxdepth:", training_scores)
    print("testing score_maxdepth:", testing_scores)
    ax.plot(maxdepths,training_scores,label="Training Score", color='r', linestyle='--')
    ax.plot(maxdepths,testing_scores,label="Testing Score", color='b')
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy Score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4,1.05)
    plt.grid(axis='y')
    plt.suptitle("Gradient Boosting Classifier")
    plt.savefig('figure4.jpg')
def test_GradientBoostingClassifier_learning(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随学习率参数的影响

    :param data:    可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    learnings=np.linspace(0.01,1.0)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    testing_scores=[]
    training_scores=[]
    for learning in learnings:
        clf=ensemble.GradientBoostingClassifier(learning_rate=learning)
        clf.fit(X_train,y_train)
        training_scores.append(clf.score(X_train,y_train))
        testing_scores.append(clf.score(X_test,y_test))
    print("training score_learing rate:", training_scores)
    print("testing score_learning rate:", testing_scores)
    ax.plot(learnings,training_scores,label="Training Score", color='r', linestyle='--')
    ax.plot(learnings,testing_scores,label="Testing Score", color='b')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy Score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4,1.05)
    plt.grid(axis='y')
    plt.suptitle("Gradient Boosting Classifier")
    plt.savefig('figure2.jpg')
def test_GradientBoostingClassifier_subsample(*data):
    '''
    测试 GradientBoostingClassifier 的预测性能随 subsample 参数的影响

    :param data:    可变参数。它是一个元组，这里要求其元素依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记
    :return:  None
    '''
    X_train,X_test,y_train,y_test=data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    subsamples=np.linspace(0.01,1.0)
    testing_scores=[]
    training_scores=[]
    for subsample in subsamples:
            clf=ensemble.GradientBoostingClassifier(subsample=subsample)
            clf.fit(X_train,y_train)
            training_scores.append(clf.score(X_train,y_train))
            testing_scores.append(clf.score(X_test,y_test))
    ax.plot(subsamples,training_scores,label="Training Score", color='r', linestyle='--')
    ax.plot(subsamples,testing_scores,label="Training Score", color='b')
    ax.set_xlabel("Subsample")
    ax.set_ylabel("Accuracy Score")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4,1.05)
    plt.grid(axis='y')
    plt.suptitle("Gradient Boosting Classifier")
    plt.savefig('figure5.jpg')

def evaluate_model(y_true,y_pred):
    print('Accuracy Score(normalize=True):', accuracy_score(y_true, y_pred, normalize=True))
    # print('Precision Score:', precision_score(y_true, y_pred,pos_label=1))
    # print('Recall Score:', recall_score(y_true, y_pred,pos_label=1))
    # print('F1 Score:', f1_score(y_true, y_pred,pos_label=1))
    print('Classification Report:\n', classification_report(y_true, y_pred,labels=[0,1],target_names=["Normal Website", "Hidden Link Website"]))
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred, labels=[0, 1]))

def test_cross_val_score(X,y):
    '''
    测试  cross_val_score 的用法

    :return: None
    '''
    num_folds = np.arange(1, 11)
    result1=cross_val_score(ensemble.GradientBoostingClassifier(n_estimators=120,max_depth=6,max_leaf_nodes=None,learning_rate=0.75,subsample=0.96),X,y,cv=10) # 使用 LinearSVC 作为分类器
    print("Cross Val Score is:",result1)
    print("Average Cross Val Score is:",sum(result1)/len(result1))

    result2 = cross_val_score(ensemble.GradientBoostingClassifier(n_estimators=120,max_depth=6,max_leaf_nodes=None,learning_rate=0.75,subsample=0.96), X, y, cv=10,scoring='f1')  # 使用 LinearSVC 作为分类器
    print("F1 Score is:", result2)
    print("Average F1 Score is:", sum(result2) / len(result2))

    result3 = cross_val_score(ensemble.GradientBoostingClassifier(n_estimators=120,max_depth=6,max_leaf_nodes=None,learning_rate=0.75,subsample=0.96), X, y, cv=10,scoring='precision')  # 使用 LinearSVC 作为分类器
    print("Precision Score is:", result3)
    print("Average Precision Score is:", sum(result3) / len(result3))

    result4 = cross_val_score(ensemble.GradientBoostingClassifier(n_estimators=120,max_depth=6,max_leaf_nodes=None,learning_rate=0.75,subsample=0.96), X, y, cv=10,
                             scoring='recall')  # 使用 LinearSVC 作为分类器
    print("Recall Score is:", result4)
    print("Average Recall Score is:", sum(result4) / len(result4))
    fig = plt.figure()
    font = {'family': 'Times New Roman', 'size': 8}
    ax = fig.add_subplot(1, 1, 1)
    plt.tick_params(labelsize=8)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.plot(num_folds, result1, label="Accuracy Score", marker='*',color='#000000',linestyle='--')
    ax.plot(num_folds, result2, label="F1 Score", marker='s',color='#000000',linestyle='--')
    ax.plot(num_folds, result3, label="Precision Score", marker='o',color='#000000',linestyle='--')
    ax.plot(num_folds, result4, label="Recall Score", marker='^',color='#000000',linestyle='--')
    ax.set_xlabel("Number of Folds",font)
    ax.set_ylabel("Test Score",font)
    ax.set_title("Gradient Boosted Decision Tree Classifier",font)
    ax.legend(framealpha=0.2, loc='best',prop=font)
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
    # y_pred, y_test=test_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
    # print(y_test)
    # print(y_pred)
    # evaluate_model(y_test, y_pred)
    # end=time.clock()
    # print(start-end)
    # test_cross_val_score(X, Y)
    y_pred, y_test =test_GradientBoostingClassifier(X_train, X_test, y_train, y_test)  # 调用 test_GradientBoostingClassifier
    # evaluate_model(y_test, y_pred)
    # end = time.clock()
    # print('runtime:',end - start)
    # test_GradientBoostingClassifier_num(X_train,X_test,y_train,y_test) # 调用 test_GradientBoostingClassifier_num
    # test_GradientBoostingClassifier_maxdepth(X_train,X_test,y_train,y_test) # 调用 test_GradientBoostingClassifier_maxdepth
    # test_GradientBoostingClassifier_learning(X_train,X_test,y_train,y_test) # 调用 test_GradientBoostingClassifier_learning
    # test_GradientBoostingClassifier_subsample(X_train,X_test,y_train,y_test) # 调用 test_GradientBoostingClassifier_subsample
