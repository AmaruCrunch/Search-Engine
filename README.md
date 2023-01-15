# Search-Engine
Information Retrieval model built on Wikipedia for our Information Retrieval class 2023' 

This project was built based on Wikipedia. The preproccessed data is available at: https://console.cloud.google.com/storage/browser/wiki_preproccess

The proccesed Indices are available at: https://console.cloud.google.com/storage/browser/wiki_irt_data

### Getting started:
follow instructions in run_frontend_in_gcp.sh to start a instance on GCP

upload search engine files to instance

!python3 search_frontend.py

### Method Overview 
1.	Data: 
-	Title, body and anchor Inverse Index both stemmed and clean
-	Document title, length, Page Rank and Page View Indices
-	Word Vectors index 
2.	Retrieval: 
-	Tokenize and stem query
-	Retrieve posting lists
-	Expand Query
-	Calculate binary score for title
-	Calculate BM25 score for text
-	Retrieve the top scoring documents for both methods
3.	Ranking: 
-	Merge results of both methods
-	Introduce Page Rank and Page view score 
-	Reorder based on the joint score
-	Return top ranking documents
4.	Evaluation
-	Manual evaluation: Do results represent good retrieval?
-	Optimize on Map@40 
5.	Noticeably features: 
-	Query expansion using Word2Vec 
-	Parallelization
-	User based - system can improve overtime
-	
![image](https://user-images.githubusercontent.com/87470704/212546884-a672b6b1-64b9-4e0f-a568-a348ec22ac8f.png)

The main search engine is based off the following logic:

![image](https://user-images.githubusercontent.com/87470704/212547089-d8f939ef-05fd-4b9b-a23b-271f90d029f2.png)
