'''
data manipulation utility
updated: 2018.12.3
'''
import numpy as np
'''
inputs: list of data required conversion in string type
totem: split totem
dtype: convert datatype
'''
def str2float1D(inputs, totem=',', dtype='float'):
    # split string
    tmp_list = []
    buf = ''
    for c in inputs:
        if c == totem:
            tmp_list.append(buf)
            buf = ''
        else:
            buf += c
    else:
        tmp_list.append(buf)
    # convert data type:
    output_list = []
    for t in tmp_list:
        try:
            output_list.append(np.array(t).astype(dtype)+0)
        except:
            output_list.append(t)
    return output_list
def str2array2D(inputs, totem='&', dtype='float'):
    # split string
    tmp_list = []
    buf = ''
    for c in inputs:
        if c == totem:
            tmp_list.append(buf)
            buf = ''
        elif c != '(' and c != ')' and c != ' ':
            buf += c
    else:
        tmp_list.append(buf)
    # convert to 2D format
    for idx, tmp in enumerate(tmp_list):
        tmp_list[idx] = str2float1D(tmp)
    return tmp_list
'''
find value from key
'''
def find(key, key_list):
    for k_idx, k in enumerate(key_list):
        if k == key:
            return k_idx
    else:
        return -1
'''
split string
'''
def str_splitter(string, cut_pts=[0]):
    output_list = []
    p1 = 0
    p2 = 0
    for c in cut_pts:
        p2 = c
        output_list.append(string[p1:p2])
        p1 = c
    else:
        output_list.append(string[p1:len(string)])
    return output_list
"""
debug
"""
if __name__ == '__main__':
    print(str2float1D('asdf,123,77,4e5'))
    print(find(1,range(2,9)))
    print(str_splitter('asdfqwer', [1,2,3]))
    