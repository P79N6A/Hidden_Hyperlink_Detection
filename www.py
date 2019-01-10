#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
from lxml import etree
import codecs
import urllib.parse
import urllib
def read_file(filename):
    try:
        f=codecs.open(filename,'r',encoding='utf-8')
        Web_data=f.readlines()
        f.close()
        Web_data = '\n'.join(Web_data)
    except:
        f=codecs.open(filename,'r','gb2312')
        Web_data = f.readlines()
        # print(Web_data)
        # print(filename)
        f.close()
        Web_data = '\n'.join(Web_data)
    return Web_data

def parse_html(Web_data):
    html=etree.HTML(Web_data)
    # div_a=html.xpath('//div/a//@href')
    # div_a_text=html.xpath('//div/a/text()')
    # print(div_a_text)
    # td_a=html.xpath('//td/a//@href')
    # td_a_text=html.xpath('//td/a/text()')
    # marquee_a=html.xpath('//marquee/a//@href')
    # marquee_a_text=html.xpath('//marquee/a/text()')
    # ul_a=html.xpath('//ul/a//@href')
    # ul_a_text=html.xpath('//ul/a/text()')
    A_text=html.xpath('//a/text()')
    # A_text=A_t.xpath('')
    A_list=html.xpath('//a//@href')
    # print(A)
    # A_list=div_a+td_a+marquee_a+ul_a
    # Text_list=div_a_text+td_a_text+marquee_a_text+ul_a_text
    print(A_text)
    print(len(A_text))
    print(len(A_list))
    # print(Text_list)
    return A_list,A_text

def get_domain(url_list):
    for url in url_list:
        parts=urllib.parse.urlparse(url)
        host=parts.netloc
        if host:
            print(host)
if __name__=='__main__':
    filename='0a6d6c2be3b9d786f17d40dc9a993db0的副本'
    Web_data=read_file(filename)
    A_list, Text_list=parse_html(Web_data)
    get_domain( A_list)

