'''
function library
updated: 2017.2.17
'''
import os
import configparser as cp
import csv

'''
Config input/output
'''
def setConfig(configPath, sec, opt, val):
    parser = cp.ConfigParser()
    parser.optxform = str
    if not os.path.isfile(os.path.join(configPath)):
        open(configPath, 'w')
    parser.read(configPath)
    if not parser.has_section(sec):
        parser.add_section(sec)
    parser.set(sec, opt, val)
    parser.write(open(configPath, 'w'))
def getConfig(configPath, sec, opt, cvt=None, totem=',', incBlank=False, incBrackets=False, isCoord=False, coordDim=2):
    parser = cp.ConfigParser()
    parser.optxform = str
    parser.read(configPath)
    string = parser.get(sec, opt)
    if cvt != None:
        string = str2array(string, totem, cvt, incBlank, incBrackets, isCoord, coordDim)
    return string
'''
data type conversion
string: input string for converting
totem: character for spliting string
convert: auto tramsform data to certain type. e.g. int, float
incBlank: consider whether blank a character
incBrackets: consider whether brackets a character
isCoord: whether input are coordinates
'''
def str2array(string, totem=',', convert=None, incBlank=False, incBrackets=False, isCoord=False, coordDim=2):
    dim_counter = 1     # used for multi-dimension with same totem
    tmpArray = []
    tmpCell = ''
    output_list = []
    for idx, cell in enumerate(string):
        if isCoord:
            ## at totem or end, save tmp value to output array
            if cell == totem or len(string) == idx+1:
                if dim_counter%coordDim == 0:      # parse tmp array to output list
                    tmpCell = datatypeConvert(tmpCell, 'float')
                    tmpArray.append(tmpCell)
                    output_list.append(tmpArray)
                    tmpArray = []
                    tmpCell = ''
                else:
                    tmpCell = datatypeConvert(tmpCell, 'float')
                    tmpArray.append(tmpCell)
                    tmpCell = ''
                dim_counter += 1
            else:
                if cell == '[' or cell == ']' or cell == ' ':
                    pass
                else:
                    tmpCell += cell
        else:
            ## at totem or end, save tmp value to output array
            if cell == totem:
                tmpCell = datatypeConvert(tmpCell, convert)
                output_list.append(tmpCell)
                tmpCell = ''
            elif len(string) == idx+1:
                if incBrackets and incBlank:
                    tmpCell += cell
                elif incBrackets:
                    if cell == ' ':
                        pass
                    else:
                        tmpCell += cell
                elif incBlank:
                    if cell == '[' or cell == ']':
                        pass
                    else:
                        tmpCell += cell
                else:
                    if cell == '[' or cell == ']' or cell == ' ':
                        pass
                    else:
                        tmpCell += cell
                tmpCell = datatypeConvert(tmpCell, convert)
                output_list.append(tmpCell)
                tmpCell = ''
            else:
                if incBrackets and incBlank:
                    tmpCell += cell
                elif incBrackets:
                    if cell == ' ':
                        pass
                    else:
                        tmpCell += cell
                elif incBlank:
                    if cell == '[' or cell == ']':
                        pass
                    else:
                        tmpCell += cell
                else:
                    if cell == '[' or cell == ']' or cell == ' ':
                        pass
                    else:
                        tmpCell += cell
    return output_list
def datatypeConvert(inputData, convertType=None):
    if convertType == 'int':
        return int(inputData)
    elif convertType == 'float':
        return float(inputData)
    else:
        #print 'no converting performed.'
        return inputData
'''
array size transformation
'''
def transMatrix2D(array):
    outputArray = []
    try:
        n = len(array[0])
        m = len(array)
        for i in range(m):
            for j in range(n):
                try:
                    outputArray[j].append(array[i][j])
                except:
                    outputArray.append([])
                    outputArray[j].append(array[i][j])
    except:
        print('transverse fail')
        outputArray = array
    return outputArray
'''
find the target in a certain array
'''
def find(array, target, tol=None, typ='array'):
    output = []
    if tol == None:     # find charactor
        for idx, cell in enumerate(array):
            if cell == target:
                if typ == 'single':
                    return idx
                else:
                    output.append(idx)
            else:
                continue
    else:               # find number
        for idx, cell in enumerate(array):
            if cell == target or cell-tol <= target <= cell+tol:
                if typ == 'single':
                    return idx
                else:
                    output.append(idx)
            else:
                continue
    return output
'''
find same element in two 1D array
'''
def findEqualCell(array1, array2):
    for idx1, cell1 in enumerate(array1):
        for cell2 in array2:
            if cell1 == cell2:
                return array1[idx1]
            else:
                continue
    return None
'''
transform to csv file
'''
def toCSV(array, fileName='csv_output.csv'):
    with open(fileName,'wb') as csvfile:
        datawriter = csv.writer(csvfile)
        for cell in array:
            datawriter.writerow(cell)
'''
testing entry
'''
if __name__ == '__main__':
    test = str2array('[1,5,32],[5,5,58],[7,49812.2,5074.5]',isCoord=True,coordDim=3)
    print(test)