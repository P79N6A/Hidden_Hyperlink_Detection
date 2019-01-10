#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
with open('file_list_20170430_new.txt', 'r') as f:
    flag_list=list()
    count_d=0
    for i in f:
        flag = i.split(',', 4)[1]
        MD5 = i.split(',', 4)[2]
        if flag == 'd':
            flag_list.append(1)
            count_d+=1
        elif flag == 'n':
            flag_list.append(0)
print(count_d)
print(len(flag_list))