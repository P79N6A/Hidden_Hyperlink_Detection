# -*- coding:utf-8 -*-
# #!/usr/bin/python
# Extract words from Web for BlackChain
# Author: Fay
# Version: 1.0
# Date: 、
import os
import re
import math
import jieba
import sys
reload(sys)
sys.setdefaultencoding( "gb2312" )
everyword=0
all_dic = {}
dic={}
finaldict={}

#对黑链文字进行分词和词频统计
cf=open('./wordsort.txt','w') 
with open('./Blackchain/Blackword.txt','r') as bf:
	#开始分词
	seg_list=jieba.cut_for_search(bf.read())
	for bword in seg_list:
		if len(bword)>1: #去掉分词长度小于1的词
			if (u'\u4e00' <= bword <= u'\u9fff'):
				if (bword in all_dic.keys()): # all_dic 训练集中所有字符的字典：key 为词， value 为出现次数
					all_dic[bword]+=1
				else:
					all_dic[bword]=1
				# print all_dic
	dic=sorted(all_dic.iteritems(),key=lambda d:d[1],reverse=True)
	# print dic
	for(key2, value2) in dic:
		finaldict[key2]=value2
		print key2,value2
		cf.write(key2+','+str(value2)+'\n')


		

