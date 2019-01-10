#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author：Janet Chou
import numpy as np
with open('vector.txt','w') as f_vector:
    a=np.array([1.333333,3.555555,4.5655656])
    for i in a:
        f_vector.write(str(i))