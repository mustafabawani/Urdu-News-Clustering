from asyncio.windows_events import NULL
import math
from sklearn.cluster import KMeans
import glob
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
def headlineReader(path):
    headlines = []
    file_paths=[]
    extraWords = ['.doc','Urdu NEWS dataset\\','voa','bbc','dataset','Entertainment','sports','miscleneous','politics','\\',"'"]
    folderPath = path
    files = [file for file in glob.glob(folderPath + "**/*.DOC", recursive=True)]
    
    for file in files:
        file_paths.append(file)
        file = re.sub(r'|'.join(map(re.escape, extraWords)), '', file)
        headlines.append(file)
    return headlines,file_paths



def docReader(path):
    fop=open(path.replace("\\\\","\\"),"r",encoding="utf-8")
    data=fop.read()
    english_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for c in data:
        if c in english_letters:
            break
    lines_before_date = data.split(c)[0].split("\n")
    lines_with_content = [line for line in lines_before_date if line.strip() != ""  ]
    second_para = '\n'.join(lines_with_content[1:])
    # second_para=list(second_para)
    return second_para

    


#####MAIN3####
headlines,file_paths=headlineReader('Urdu NEWS dataset//')
# print(headlines[0])
second_para=docReader(file_paths[0])
# print(second_para)

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
transformed_documents = vectorizer.fit_transform(headlines)
# print(transformed_documents)

vocab=vectorizer.get_feature_names_out()
# print(vocab)

num_clusters = 181
num_seeds = 10
max_iterations = 300

pca_num_components = 2
tsne_num_components = 2


labels_color_map = {}

for i in range(num_clusters):
    labels_color_map[i]= "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
# print(labels_color_map)
clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations
    )


labels = clustering_model.fit_predict(transformed_documents)
# print(len(labels))
X = transformed_documents.todense()
list_dense=X.tolist()
# print(list_dense[0])
dp=pd.DataFrame(list_dense,columns=vocab)

centroids  = clustering_model.cluster_centers_
centroid_labels = [centroids[i] for i in labels]
print((centroids))
# print(centroid_labels)

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(labels):
    clustered_sentences[cluster_id].append(headlines[sentence_id])



def vectorDotProduct(v1,v2):
    dp=0
    for i in range(0,len(v1)):
            dp=dp+(v1[i]*v2[i])
    return(dp)


def vectorMagnitude(v):
    m=0
    for i in range(0,len(v)):
            m=m+(v[i]**2)
    return(math.sqrt(m))

#calculating consine similarity btw document and query vector
def cosineScore(v1,v2):
    cs=vectorDotProduct(v1,v2)
    cs=cs/(vectorMagnitude(v1))
    cs=cs/(vectorMagnitude(v2))
    return cs

def queryProcessing(q,vocab,centroids,list_dense,headlines,clustered_sentences):
    
    q=q.split(" ")
    queryVector=[]
    
    for word in vocab:
        if word in q:
            queryVector.append(1)
            # print("IN")
        else:
            queryVector.append(0)
    for i,word in enumerate(vocab):
        if word in q:
            queryVector[i]=(1+math.log10(queryVector[i]))

    max_cluster_index=MostSimilarCentroid(queryVector,vocab,centroids)
    docs=clustered_sentences[max_cluster_index]
    # print(len(docs))
    documents_index=[]
    for d in docs:
        documents_index.append(headlines.index(d))

    ranking=[]
    for idx in documents_index:      
        document_vector=(list_dense[idx])
        ranking.append(cosineScore(document_vector,queryVector))

    ranked_doc=[]
    while ranking:
        max = ranking[0]  
        for x in ranking: 
            if x > max:
                max = x
        ind=ranking.index(max)
        ranked_doc.append(docs[ind])
        ranking.remove(max)    
    # print('a')
    return ranked_doc



def MostSimilarCentroid(queryVector,vocab,centroids):    
    centroid_scores=[]
    for centroid in centroids:
        cs=cosineScore(centroid,queryVector)
        centroid_scores.append(cs)
    # print(centroid_scores)
    max_score=max(centroid_scores)
    max_cluster_index=centroid_scores.index(max_score)
    return(max_cluster_index)

def clustering_process(query):
    ranked_doc=queryProcessing(query,vocab,centroids,list_dense,headlines,clustered_sentences)
    # print(ranked_doc)
    ranked_index=[]
    for i in range(len(ranked_doc)):
        for j in range(len(headlines)):
            if(i==j):
                ranked_index.append(j)

    # khulasa=[]#"درہ آدم خیل میں فاٹا کی پہلی یونیورسٹی
    # for i in ranked_index:
    #     second_para=docReader(file_paths[i])
    # khulasa.append(second_para)
    return ranked_doc
    # print(khulasa)



# print(dp.head)