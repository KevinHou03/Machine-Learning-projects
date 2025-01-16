from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_texts = [" ".join(text) for text in train_df['tokenized_speech'].tolist()]
dev_texts = [" ".join(text) for text in dev_df['tokenized_speech'].tolist()]


def vocab_parameter(sizes, train_texts, dev_texts, train_Xb, dev_Xb, train_y, dev_y):
    results = []

    for vocab_size in sizes:        

        tfidf_vectorizer = TfidfVectorizer(max_features=vocab_size)
        X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
        X_dev_tfidf = tfidf_vectorizer.transform(dev_texts)


        train_combined = np.hstack((train_Xb, X_train_tfidf.toarray()))
        dev_combined = np.hstack((dev_Xb, X_dev_tfidf.toarray()))


        classifier = LogisticRegression()
        classifier.fit(train_combined, train_y)


        dev_pred = classifier.predict(dev_combined)
        performance = accuracy_score(dev_y, dev_pred)


        results.append({
            'vocab_size': vocab_size,
            'performance': performance
        })


    return pd.DataFrame(results)

sizes = [1000, 2000, 5000]
results_df = vocab_parameter(sizes, train_texts, dev_texts, train_Xb, dev_Xb, train_y, dev_y)

plt.figure(figsize=(8, 6))
plt.plot(results_df['vocab_size'], results_df['performance'], marker='o')
plt.xlabel('Vocabulary Size')
plt.ylabel('Accuracy on Development Set')
plt.title('Effect of Vocabulary Size on Classifier Performance')
plt.grid(True)
plt.show()


def topological_sort(DAG):
    pass

'''
def countPaths(DAG):
    Order = topological_sort(DAG)
    Count = [0 for temp_vertex in range(len(DAG.vertices))]
    Count[start_vertex] = 1

    for item in topoOrder:
        for vertex in DAG.adjacent(item):
            pathCount[vertex] += pathCount[item]
    
    return sum(Count)

def topological_sort(DAG):
    pass
'''
