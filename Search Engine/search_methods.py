from math import log, sqrt
from typing import List, Tuple, Dict
from collections import Counter


def tfidf_vectors(inverted_index: Dict[str, List[Tuple[int, int]]],
                  df: Dict[str, int],
                  Q: List[str],
                  DL: Dict[int, int]) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Calculate the tfidf vectors for a query and for all candidate documents
    Returns a dict for query and a dict for each doc in documents
    """
    N = len(DL)
    query_vec = {}
    doc_vecs = {}
    # calculate the tf-idf score for each word in the query
    for word in Q:
        tf = Q.count(word) / len(Q)
        idf = log(N/df.get(word, 1)) #.get(word, 1))
        query_vec[word] = tf * idf
    # Only calculate the tf-idf score for each word in the document if it's in the query
    for word in Q:
        for doc_id, tf in inverted_index.get(word, []):
            if doc_id not in doc_vecs:
                doc_vecs[doc_id] = {}
            try:
                tf_norm = tf / DL[doc_id]
                idf = log(N/df[word])
            except ZeroDivisionError:
                pass
            doc_vecs[doc_id][word] = tf_norm * idf
    return query_vec, doc_vecs

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute the cosine similarity between two vectors
    """
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1.keys()).union(vec2.keys()))
    norm1 = sqrt(sum(val ** 2 for val in vec1.values()))
    norm2 = sqrt(sum(val ** 2 for val in vec2.values()))
    return dot_product / (norm1 * norm2)

def tfidf(inverted_index: Dict[str, List[Tuple[int, int]]],
           df: Dict[str, int],
           Q: List[str],
           DL: Dict[int, int],
           limit=100) -> List[Tuple[int, float]]:
    """
    
    """       
    query_vec, doc_vecs = tfidf_vectors(inverted_index, df, Q, DL)
    # compute the similarity between the query and each document
    similarities = [(doc_id, cosine_similarity(query_vec, doc_vec)) for doc_id, doc_vec in doc_vecs.items()]
    # sort the similarities in descending order and return the top 100 closest documents
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:limit]


"""
BM25
"""
def bm25(inverted_index: Dict[str, List[Tuple[int, int]]],
                df: Dict[str, int],
                Q: List[str],
                DL: Dict[int, int],
                avdl: float,
                N: int,
                k1=1.2, b=0.75, limit=100) -> Dict[int,float]:

    scores = Counter()
    for word in Q:
        for doc_id, tf in inverted_index.get(word, []):
            dl = DL[doc_id]
            n = df[word]
            tf_weight = ( (k1+1) * tf ) / ( k1 * ( (1-b) + b * dl / avdl ) + tf )
            idf_weight = log( (N-n+0.5) / (n+0.5) )
            scores[doc_id] += tf_weight * idf_weight
    
    
    return scores.most_common(limit)

"""
BINARY
"""
