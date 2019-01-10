#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
from lxml import etree
import urllib.parse
import urllib
import os
import codecs
def traverse_directory(WebDirectory,mainfile):
    Dark_MD5=list()
    with open(mainfile,'r') as f:
        for i in f:
            flag=i.split(',',4)[1]
            MD5=i.split(',',4)[2]
            if flag=='d':
                each_file=os.path.join(WebDirectory,MD5)
                Dark_MD5.append(each_file)
    return Dark_MD5


def read_file(filename):
    try:
        f=codecs.open(filename,'r',encoding='utf-8')
        Web_data=f.readlines()
        # print(filename)
        print('******')
        f.close()
        Web_data = '\n'.join(Web_data)
    except:
        f=codecs.open(filename,'r','gb2312')
        Web_data = f.readlines()
        # print(Web_data)
        # print(filename)
        print('******')
        f.close()
        Web_data = '\n'.join(Web_data)
    return Web_data

def parse_html(Web_data):
    html=etree.HTML(Web_data)
    div_a=html.xpath('//div/a//@href')
    div_a_text=html.xpath('//div/a/text()')
    # print(div_a_text)
    td_a=html.xpath('//td/a//@href')
    td_a_text=html.xpath('//td/a/text()')
    marquee_a=html.xpath('//marquee/a//@href')
    marquee_a_text=html.xpath('//marquee/a/text()')
    ul_a=html.xpath('//ul/a//@href')
    ul_a_text=html.xpath('//ul/a/text()')
    # A_text=html.xpath('//a/text()')
    # A_list=html.xpath('//a//@href')
    # print(A)
    A_list=div_a+td_a+marquee_a+ul_a
    Text_list=div_a_text+td_a_text+marquee_a_text+ul_a_text
    # print(Ast)
    # print(Text_list)
    return A_list,Text_list

def get_domain(url_list,f_A):
    for url in url_list:
        parts=urllib.parse.urlparse(url)
        host=parts.netloc
        if host:
            f_A.write(host+'\n')




if __name__=='__main__':
    try:
        f_text=open('Black_text.txt','w')
        f_A=open('Black_A.txt','w')
        mainfile='file_list_20170430_new.txt'
        WebDirectory='./file/'
        Dark_MD5=traverse_directory(WebDirectory,mainfile)
        print(len(Dark_MD5))
        for aline in Dark_MD5:
            Web_data=read_file(aline)
            url_list,text_list=parse_html(Web_data)
            for vocabulary in text_list:
                f_text.write(vocabulary+'\n')
            get_domain(url_list,f_A)
    except:
        print(aline)