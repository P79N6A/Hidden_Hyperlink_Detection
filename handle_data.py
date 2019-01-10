#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
# Black_vocabulary_list=list()
# with open('Black_text.txt','r') as f:
#     for i in f:
#         Black_vocabulary_list.append(i.strip('\n'))
# Black_vocabulary_list.sort()
# print(Black_vocabulary_list)
#
#
# with open('Text_Black.txt','w') as f2:
#     for j in Black_vocabulary_list:
#         f2.write(j+'\n')
from bs4 import BeautifulSoup
import numpy as np
import urllib.request
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
with open('0a6d6c2be3b9d786f17d40dc9a993db0的副本','r',encoding='utf-8') as f:
    content=f.readlines()
    content = '\n'.join(content)
    soup=BeautifulSoup(content,"html.parser")
    d_fea_list = [content.count(fea) for fea in D_Feature]
    d_t1 = np.asarray(d_fea_list)
    d_fea_np = np.hstack((d_t1, [d_t1.sum(), d_t1.max(), d_t1.std(), d_t1.mean(), sum(d_t1 > 0)]))
    print(d_fea_np)
    print(len(d_fea_np))






# Black_vocabulary_list=set()

# with open('Black_text.txt','r') as f:
#     for i in f:
#         Black_vocabulary_list.add(i)
# with open('text_Black.txt','w') as f2:
#     for j in Black_vocabulary_list:
#         f2.write(j)