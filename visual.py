from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import glob
import re
import random

def headlineReader(path):
    headlines = []
    extraWords = ['.doc','Urdu NEWS dataset\\','voa','bbc','dataset','Entertainment','sports','miscleneous','politics','\\',"'"]
    folderPath = path
    files = [file for file in glob.glob(folderPath + "**/*.DOC", recursive=True)]
    
    for file in files:
        file = re.sub(r'|'.join(map(re.escape, extraWords)), '', file)
        headlines.append(file)
    return headlines


headlines=headlineReader('Entertainment//')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(headlines)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)