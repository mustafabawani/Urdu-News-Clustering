from asyncio.windows_events import NULL
from locale import normalize
from pathlib import Path
from numpy import absolute, empty
from sklearn.cluster import KMeans, k_means
import glob
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

def headlineReader(path):
    headlines = []
    extraWords = ['.doc','Urdu NEWS dataset\\','voa','bbc','dataset','Entertainment','sports','miscleneous','politics','\\',"'"]
    folderPath = path
    files = [file for file in glob.glob(folderPath + "**/*.DOC", recursive=True)]
    
    for file in files:
        file = re.sub(r'|'.join(map(re.escape, extraWords)), '', file)
        headlines.append(file)
    return headlines

#####MAIN3####
headlines=headlineReader('Urdu NEWS dataset//')

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
transformed_documents = vectorizer.fit_transform(headlines)


num_clusters = 181
num_seeds = 10
max_iterations = 300

pca_num_components = 2
tsne_num_components = 2

Sum_of_squared_distances = []
silhouette_avg = []
K = range(350,400)
for num_clusters in K :
    clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations
    )
    clustering_model.fit(transformed_documents)
    # cluster_labels = clustering_model.labels_
    Sum_of_squared_distances.append(clustering_model.inertia_)
    print(num_clusters)
plt.plot(K,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# labels_color_map = {}

# for i in range(num_clusters):
#     labels_color_map[i]= "#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])
# print(labels_color_map)
# clustering_model = KMeans(
#         n_clusters=num_clusters,
#         max_iter=max_iterations
#     )

# labels = clustering_model.fit_predict(transformed_documents)  
# X = transformed_documents.todense()

# reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
# print reduced_data

# fig, ax = plt.subplots()
# for index, instance in enumerate(reduced_data):
#     # print instance, index, labels[index]
#     pca_comp_1, pca_comp_2 = reduced_data[index]
#     color = labels_color_map[labels[index]]
#     ax.scatter(pca_comp_1, pca_comp_2, c=color)
# plt.show()



# print labels



# embeddings = TSNE(n_components=tsne_num_components)
# Y = embeddings.fit_transform(X)
# plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
# plt.show()
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# pca=PCA(n_components=2)
# plot_point=pca.fit_transform(transformed_documents.toarray())
# x_axis=[o[0] for o in plot_point]
# y_axis=[o[1] for o in plot_point]
# fig, ax = plt.subplots(figsize=(50,50))
# ax.scatter(x_axis,y_axis)

# plt.savefig("trc.png")

# transformed_documents_as_array = transformed_documents.toarray()
# import pandas as pd

# # make the output folder if it doesn't already exist
# Path("./tf_idf_output").mkdir(parents=True, exist_ok=True)

# # construct a list of output file paths using the previous list of text files the relative path for tf_idf_output
# output_filenames = [str(txt_file).replace(".txt", ".csv").replace("txt/", "tf_idf_output/") for txt_file in headlines]

# # loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
# for counter, doc in enumerate(transformed_documents_as_array):
#     # construct a dataframe
#     tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
#     one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)

#     # output to a csv using the enumerated value for the filename
#     one_doc_as_df.to_csv(output_filenames[counter])

# print(transformed_documents)
# # print(headlines)



# print(dp)
# reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
# fig, ax = plt.subplots()
# for index, instance in enumerate(reduced_data):
#     pca_comp_1, pca_comp_2 = reduced_data[index]
#     color = labels_color_map[labels[index]]
#     ax.scatter(pca_comp_1, pca_comp_2, c=color)
# plt.show()



#################################
###centroid k sare words#######
############################

# names = []
# for vector in centroid_labels:
#     name = []
#     for word, value in zip(vocab, vector):
#         if value > 0:
#             name.append(word)
#     names.append(name)
#     print(name)
    