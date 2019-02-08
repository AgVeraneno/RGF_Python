import numpy as np

def find(key, key_list):
    for k_idx, k in enumerate(key_list):
        if key == k:
            return k_idx
        else:
            continue
    else:
        return -1
def mfind(key, key_list):
    output_list = []
    for k_idx, k in enumerate(key_list):
        if key == k:
            output_list.append(k_idx)
        else:
            continue
    else:
        return output_list
def string_splitter(string, totem=',', nobracket=False, nospace=False):
    forbid_list_a = ['(',')','[',']','{','}']
    forbid_list_b = [' ']
    output = []
    tmp_str = ''
    for char in string:
        if char == totem:
            output.append(tmp_str)
            tmp_str = ''
        elif (nobracket and char in forbid_list_a) or \
             (nospace and char in forbid_list_b):
            continue
        else:
            tmp_str += char
    else:
        output.append(tmp_str)
        return output
def convert_datatype_1D(data_list, datatype='float'):
    output = []
    for data in data_list:
        try:
            output.append(np.array(data).astype(datatype))
        except:
            output.append(np.array(data))
    else:
        return output
def convert_datatype_2D(data_list, datatype='float'):
    output = []
    for data in data_list:
        output.append([])
        try:
            output[-1].append(np.array(data).astype(datatype))
        except:
            for subdata in data:
                try:
                    output[-1].append(np.array(subdata).astype(datatype))
                except:
                    output[-1].append(np.array(subdata))
    else:
        return output
    
if __name__ == '__main__':
    output = string_splitter('asdf,456,789,55 5,[weroi]', ',', False,True)
    print(output)
    output = convert_datatype_2D([['123','111'],['asdf','115']], 'complex')
    print(output)
    