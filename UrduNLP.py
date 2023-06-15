from encodings import utf_8
# from locale import normalize
from pathlib import Path
import random
import re
from sqlite3 import Date
# import urduhack
# from urduhack.tokenization import word_tokenizer
import urduhack.normalization as UN
import urduhack.preprocessing as UP
import glob

import warnings
  
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec
def FileReader(path):
    file_name = []
    extraWords = ['.doc','Urdu NEWS dataset\\','voa','bbc','dataset','entertainment','sports','miscleneous','politics','\\',"'"]
    folderPath = path
    files = [file for file in glob.glob(path+"**/*.doc", recursive=True)]
    
    for file in files:
        
        # docReader(path,file)
        file = re.sub(r'|'.join(map(re.escape, extraWords)), '', file)
        file_name.append(file)
        # break
    return file_name


###########################################################
##                                                       ##       
##          Static search snippets                       ##
##         using to show short summary for each          ##
##                                                       ##
###########################################################

# def staticSearchSnippets(path,)
def docReader(path,file_name):
    fop=open(file_name.replace("\\\\","\\"),"r",encoding="utf-8")
    data=fop.read()
    english_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for c in data:
        if c in english_letters:
            break

    lines_before_date = data.split(c)[0].split("\n")
    lines_with_content = [line for line in lines_before_date if line.strip() != ""]
    second_para = "\n".join(lines_with_content[1:])
    print(second_para)





###########################################################
##                                                       ##       
##          WORD2VECTOR FUNCTIONS                        ##
##          HELPS IN CONTEXTUAL REPRESENTATION           ##
##                                                       ##
###########################################################


def stopWords():
    stop_words=[]
    f=open("Stopwords-ur.txt",encoding="utf_8")
    #make an array of all stop words
    for word in f.read().split("\n")[0:]:
        if word:
            stop_words.append(word.strip())
    # print(stop_words)
    # print(len(stop_words))
    return stop_words


def normalization(word):
    word = UN.normalize(word)
    word=UP.remove_accents(word)
    word=UP.remove_punctuation(word)
    word= UP.replace_currency_symbols(word)
    word= UP.replace_phone_numbers(word)
    word= UP.normalize_whitespace(word)
    return word

def WordTokenizer(data_set):
    tokens=[]
    stop_words=stopWords()
    i=0
    for sent in data_set:
        sent=normalization(sent)
        tokens.append(sent.split())
    # print(tokens)
    return tokens
     

def word2vecFunc(tokens):
    w2v_model=Word2Vec(min_count=2,window=10,workers=10)
    w2v_model.build_vocab(tokens,progress_per=1000)
    # print(w2v_model.corpus_count)
    # print(w2v_model.epochs)
    w2v_model.train(tokens,total_examples=w2v_model.corpus_count,epochs=w2v_model.epochs)
    print(w2v_model.wv.most_similar("گاڑیاں"))
    # sim = w2v_model.wv.n_similarity(query_tokens, sent_token)
    return w2v_model


###########################################################
##                                                       ##       
##                     Document                          ##
##                     Cluster                           ##
##                                                       ##
###########################################################



def findCentroids(cluster):
    selected_doc={}
    for key in cluster:
        if(len(cluster[key])==0):
            selected_doc[key]=0
        else:
            mid=(len(cluster[key]))/2
            mid=int(mid)
            selected_doc[cluster[key][mid]]=0
    return selected_doc



def getCluster(selected_doc,data_set):
    similarity_score={}
    # print(selected_doc_num)
    for key in selected_doc:
        dic={}
        for k in range(0,len(data_set)):
            if(key==k ):
                continue
            if(k in selected_doc.keys()):
                continue
            score=similarityScore(data_set[k],selected_doc[key])
            # print(data_set[k])
            # print(selected_doc[key])
            dic[k]=score
        similarity_score[key]=dic
    minimum={}
    cluster={}
    for key in selected_doc:
        cluster[key]=[]
    for i in range(len(data_set)):
        minimum={}
        for keys in similarity_score:
            if i in similarity_score[keys]:
                minimum[keys]=similarity_score[keys][i]
        
        if(len(minimum)==0):
            continue
        min_num=min(minimum.values())
        for key in minimum:
            if(minimum[key]==min_num):
                cluster[key].append(i)
                break
    return cluster



def similarityScore(doc1,doc2):
    sti=len(doc1)
    stj=len(doc2)
    sij=0
    mij=0

    for i in range(sti):
        for j in range(stj):
            if(doc1[i]==doc2[j]):
                mij=mij+1
    
    if(mij>0):
        avg = (sti+stj)/2
        sij=mij/avg
    
    return sij


def Kmeans(data_set):
    similarity=[]
    selected_doc={}
    for i in range(50):
        num=random.randint(0,len(data_set)-1)
        selected_doc[num]=data_set[num]
    
    cluster=getCluster(selected_doc,data_set)
    for _ in range(0,100):        
        selected_doc=findCentroids(cluster)
        for key in selected_doc:
            selected_doc[key]=data_set[key]
        cluster=getCluster(selected_doc,data_set)
    
    print(selected_doc)
    return cluster



###########################################################
##                                                       ##       
##                     Query                             ##
##                    Process                            ##
##                                                       ##
###########################################################


def queryProcess(query,data_set,cluster_index):
    query_words=[]
    centroid_docs=[]
    query_words=query.split()
    # print(query_words)
    for i in (cluster_index.keys()):
        centroid_docs.append(data_set[i])
    centroid_tokens=WordTokenizer(centroid_docs)
    cluster_score=mostSimilarClusterUsingCentroids(query_words,centroid_tokens)
    print(cluster_score)
    max_score=max(cluster_score)
    max_cluster_index=cluster_score.index(max_score)
    print(cluster_score)
    # print(cluster_index[1375])
    # for idx in (cluster_index[1375]):
    #     print(data_set[i])

def mostSimilarClusterUsingCentroids(query_tokens,centroid_tokens):
    cluster_score=[]
    sent_token=[]
    for sent_token in enumerate(centroid_tokens):
        sim = similarityScore(query_tokens, centroid_tokens[1])
        cluster_score.append(sim)
    
    return (cluster_score)

###########################################################
##                                                       ##       
##                      MAIN                             ##
##                    FUNCTION                           ##
##                                                       ##
###########################################################

data_set=FileReader('Urdu NEWS dataset\\')
tokens=WordTokenizer(data_set)
# print(tokens)

# word_2_vec=word2vecFunc(tokens)

cluster_index=Kmeans(tokens)
print(cluster_index)
queryProcess(" ٹيکنالوجي کي دنيا",data_set,cluster_index)

# print(len(tokens))

