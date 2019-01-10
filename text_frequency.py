#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
text=dict()

# print(data)
with open('Unique_BlackA.txt','r') as f1:
    for i in f1:
        i=i.strip('\n')
        freqlist = list()
        with open('Black_A没去重.txt', 'r') as f:
            for j in f:
                freq=j.count(i)
                freqlist.append(freq)
        sumtext=sum(freqlist)
        text[i]=sumtext
print(text)
print(sorted(text.items(),key=lambda item:item[1],reverse=True))


