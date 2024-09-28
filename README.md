# EX-06 Information Retrieval Using Vector Space Model in Python
### AIM: 
To implement Information Retrieval Using Vector Space Model in Python. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**DATE: 29.09.2024**
### Description: 
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix,calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and sklearn to demonstrate Information Retrieval using the Vector Space Model.
### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.
### Program:
**Importing Libraries**
```Python
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
**Sample documents stored in a dictionary**
```Python
documents = ["Shipment of gold damaged in a fire.",
             "Delivery of silver arrived in a silver truck.",
             "Shipment of gold arrived in a truck.",]
```
**Preprocessing function to tokenize and remove stopwords/punctuation**
```Python
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english")
              and token not in string.punctuation]
    return " ".join(tokens)
```
**Preprocess documents and store them in a dictionary**
```Python
preprocessed_docs = [preprocess_text(doc) for doc in documents]
```
**Construct TF-IDF matrix**
```Python
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs)
```
**Calculate cosine similarity between query and documents**
```Python
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])
    # Calculate cosine similarity between query and documents
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    # Sort documents based on similarity scores
    sorted_indexes = similarity_scores.argsort()[0][::-1]
    # Return sorted documents along with their similarity scores
    results = [(documents[i], similarity_scores[0, i]) for i in sorted_indexes]
    return results
```
**Get input from user**
```Python
query =input("Enter query: ")
```
**Perform search**
```Python
search_results = search(query, tfidf_matrix, tfidf_vectorizer)
```
**Display search results**
```Python
i=1
for result in search_results:
    print("----------------------")
    print("Rank: ",i)
    print("Document:", result[0])
    print("Similarity Score:", result[1])
    i=i+1
```
### Output:
![image](https://github.com/user-attachments/assets/84a22da8-83d2-4bd3-9a1a-5146a1fb3283)
![image](https://github.com/user-attachments/assets/eb10e5c7-299a-4f60-8c50-c834e3efae11)

### Result:
Thus, the implementation of Information Retrieval Using Vector Space Model in Python is executed successfully.
<br>
**Developed By: ROHIT JAIN D - 212222230120**
