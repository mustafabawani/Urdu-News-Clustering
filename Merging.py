import os
import math
import numpy as np
import string
import pandas as pd
Stopword=set()
import docx2txt
DataSet=pd.DataFrame
def Preprocess(line):
    for i in string.punctuation:
        if len(line.split(i)) > 1:
            line=line.replace(i,' ' )
    return line
def FolderMaking():
    import os
    import re
    import shutil
    from docx import Document
    def DataSet(src, dest, class_name):
        src_files = os.listdir(src)
        i = 0
        for file_name in src_files:
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest)
                dst_file = os.path.join(dest, file_name)
                new_file_name = class_name + "_" + str(len(os.listdir(dest))) + ".doc"
                # groundTruth.append(class_name)
                new_dst_file_name = os.path.join(dest, new_file_name)
                os.rename(dst_file, new_dst_file_name)
            i = i + 1

    path=os.getcwd()
    path=path+'/Urdu NEWS dataset/'
    os.chdir(path)
    if os.path.exists('DataSet10') and os.path.exists('TestingFolder101'):
        shutil.rmtree('DataSet10')
        shutil.rmtree('TestingFolder101')
    path = 'DataSet10/'
    path11 = 'TestingFolder101/'
    path1 = 'Urdu NEWS dataset/bbc dataset/'
    path2 = 'Urdu NEWS dataset/voa dataset/'
    #### condition to chk if files already created or not if created just load them
    count = 0
    if (os.path.exists(path) == False):
        count += 1
        os.mkdir(path)
        os.mkdir(path11)
        groundTruth = []
        class_list = ['entertainment', "miscleneous", "politics", "sports"]
        class_list1 = ['entertainment', "misc", "politics", "sports"]
        for i in range(len(class_list)):
            src = str(path1 + class_list[i])
            DataSet(src, path, class_list[i])
            src = path2 + class_list1[i]
            DataSet(src, path, class_list[i])

    # path = 'DataSet'
    direct = os.listdir(path)
    count = 0
    for i in sorted(direct):
        document = Document()
        myfile = open('DataSet10/' + i, errors='ignore').read()
        mpa = dict.fromkeys(range(32))
        myfile = myfile.translate(mpa)
        p = document.add_paragraph(myfile)
        i = i.split('.')

        i = i[0]
        document.save('TestingFolder101/' + i + '.docx')
def DataSetMaking():
    with open('StopWords', 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip(' ')
            line = line.lstrip(' ')
            Stopword.add(line)
    dir = "TestingFolder101/"
    Words = set()
    DocNames = set()
    DocCounter = 0
    print('Start')
    keepDict = {}
    # if word not in Words and len(word) != 1 and not (word.isnumeric()) and word not in Stopword:
    for file in sorted(os.listdir(dir)):
        if file.endswith(".docx") and file != '.docx':
            nameClass = file.split('_')
            nameClass = nameClass[0]
            dir1 = dir + file
            DocNames.add(file)
            result = docx2txt.process(dir1)
            for word in result.split():
                if len(word) != 1 and not (word.isnumeric()) and word not in Stopword:
                    Words.add(word)
                    if nameClass not in keepDict.keys():
                        keepDict[nameClass] = {}
                    if word not in keepDict[nameClass].keys():
                        keepDict[nameClass][word] = 0
                    keepDict[nameClass][word] += 1
        if file != '.docx':
            DocCounter += 1

    WordsToUse = Words
    SetWords = Words.copy()
    Words = list(Words)
    Words = ['Class'] + Words
    DocNames = list(DocNames)
    DocNames = sorted(DocNames)
    print(len(WordsToUse))
    WordsDict = {}
    for i in range(len(Words)):
        WordsDict.setdefault(Words[i], i)
    DocNamesDict = {}
    for i in range(len(DocNames)):
        DocNamesDict.setdefault(DocNames[i], i)
    keepSet = set()
    for words in WordsToUse:
        Flag = 0
        for ClassValue in keepDict.keys():
            if words == 'Class':
                break
            if words in keepDict[ClassValue]:
                if keepDict[ClassValue][words] > 6:
                    Flag = 1
                    break
        if Flag == 1:
            keepSet.add(words)
    keepSetList = list(keepSet)
    keepSetList = ['Class'] + keepSetList
    keepSetDict = {}
    for i in range(len(keepSetList)):
        keepSetDict.setdefault(keepSetList[i], i)

    Matrix = np.zeros([len(DocNames), len(keepSetList)], dtype=int)
    print('DoneFeatureSelection')
    Counter = 0
    TrackOfWords = set()
    ClassNames = ['entertainment', 'miscleneous', 'politics', 'sports']
    for file in sorted(os.listdir(dir)):
        if file.endswith(".docx") and file != ".docx":
            nameClass = file.split('_')
            nameClass = nameClass[0]
            dir1 = dir + file
            name = DocNames[Counter]
            Counter += 1
            result = docx2txt.process(dir1)
            for word in result.split():
                if word in keepSet and not (word.isnumeric()):
                    getrow = DocNamesDict[name]  # DocNames.index(name)
                    getcol = keepSetDict[word]  # Words.index(word)
                    Matrix[getrow][getcol] = 1
                    getClassRow = DocNamesDict[name]  # DocNames.index(name)
                    getClassCol = keepSetDict['Class']  # Words.index('Class')
                    Matrix[getClassRow][getClassCol] = ClassNames.index(nameClass)

    print('FinalDone')
    # print(Matrix)
    import pandas as pd
    DataSet = pd.DataFrame(Matrix, columns=keepSetDict.keys(), index=DocNames)
    DataSet.sort_index(inplace=True)
    DataSet.to_csv('WholeDataTesting.csv', index=True)
    print('Finish')

import pandas as pd
import math
import numpy as np
def ChangeToDic(List):
    Dict = {}
    for i in range(len(List)):
        Dict.setdefault(List[i], i)
    return Dict
def TFIDF(Data):
    DataCopy=Data.copy()
    Chcker=0
    for ColName, Col in Data.iteritems():
        Chcker+=1
        Counter = 0
        List=[]
        Flag=0
        for value in Col:
            if ColName != 'Class':
                if value != 0:
                    if Flag==0:
                        EachValue = math.log((len(DataCopy)) / (sum(DataCopy[ColName])))
                        Flag=1
                    #EachValue=1
                    List.append(EachValue)
                    #Data = Data.assign(B=List)
                    #Data.loc[Counter, ColName] = EachValue
                    #Data.at[Counter,ColName]= 20
                else:
                    List.append(0)

            Counter += 1
        if ColName!='Class':
            Data[ColName]=np.array(List)
    return Data
def TFIDFMaking():
    DataSet=pd.read_csv('WholeDataTesting.csv',index_col=0)
    print(DataSet)
    print("Start TFIDF Calculation. It will take a minute or two")
    DataSet = TFIDF(DataSet)
    DataSet.to_csv('WholeDataTFIDFTesting.csv', index=False)
    print('Exit')