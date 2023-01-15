import numpy as np
from inverted_index_gcp import InvertedIndex
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import re
from search_methods import tfidf, bm25
from nltk.stem import PorterStemmer
import concurrent.futures
from operator import itemgetter
from gensim.models import KeyedVectors



"""
Indexes AND PATHS
"""
PROJECT = ' irt-hw3'
BUCKET_NAME = 'wiki_irt_data'
PAGE_RANK_PATH = 'page_rank'
PAGE_VIEW_PATH = 'page_views'
DOC_TITLES_PATH = 'doc_titles'
DOC_LENGTH_PATH = 'doc_length'
TEXT_INDEX_PATH = 'text_index'
TEXT_BINS = 'text_bins'
TITLE_INDEX_PATH = 'title_index'
TITLE_BINS = 'title_bins'
ANCHOR_INDEX_PATH = 'anchor_index'
ANCHOR_BINS = 'anchor_bins'
STEM_TITLE_INDEX_PATH = 'stemmed_title_index'
STEM_TITLE_BINS =  'stemmed_title_bins'
STEM_BODY_INDEX_PATH = 'stemmed_text_index'
STEM_BODY_BINS = 'stemmed_text_bins'
WORD2VEC = "word2vec.model"

CORPUS_STOP_WORDS = ["category", "references", "also", "external", "links",
                     "may", "first", "see", "history", "people", "one", "two",
                     "part", "thumb", "including", "second", "following",
                     "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)



class SearchMaster:
    def __init__(self) -> None:
        print('reading indices to memory...', end=' ')
        self.page_rank = InvertedIndex.read_index(PROJECT, BUCKET_NAME, PAGE_RANK_PATH)  # dict {id:float}
        self.page_views = InvertedIndex.read_index(PROJECT, BUCKET_NAME, PAGE_VIEW_PATH)  # dict {id:int}
        self.title_index = InvertedIndex.read_index(PROJECT, BUCKET_NAME, TITLE_INDEX_PATH, TITLE_BINS)  # II
        self.body_index = InvertedIndex.read_index(PROJECT, BUCKET_NAME, TEXT_INDEX_PATH, TEXT_BINS)  # II
        self.anchor_index = InvertedIndex.read_index(PROJECT, BUCKET_NAME, ANCHOR_INDEX_PATH, ANCHOR_BINS)  # II
        self.stem_title_index = InvertedIndex.read_index(PROJECT, BUCKET_NAME, STEM_TITLE_INDEX_PATH, STEM_TITLE_BINS)  # II
        self.stem_body_index = InvertedIndex.read_index(PROJECT, BUCKET_NAME, STEM_BODY_INDEX_PATH, STEM_BODY_BINS) 
        self.stop_words = self.get_stop_words(CORPUS_STOP_WORDS)  # list
        self.titles = InvertedIndex.read_index(PROJECT, BUCKET_NAME, DOC_TITLES_PATH)  # dict {id:title}
        self.doc_length = InvertedIndex.read_index(PROJECT, BUCKET_NAME, DOC_LENGTH_PATH)  # dict {id:int}
        self.avdl = sum(self.doc_length.values()) / len(self.doc_length)
        self.N = len(self.titles)
        self.stemmer = PorterStemmer()
        self.word_vectors = KeyedVectors.load(f'{BUCKET_NAME}/{WORD2VEC}')
        self.max_pr = max(self.page_rank.values())
        self.max_pv = max(self.page_views.values())
        self.weights = [0.01, 0.01, 0.0001, 1]
        self.topn = 10
        print('Done!')


    @staticmethod
    def get_stop_words(corpus_stopwords):
        """
        Retrieves english stop words from system.
        Returns a joint list of all stop words.
        """
        english_stopwords = frozenset(stopwords.words('english'))
        all_stopwords = english_stopwords.union(corpus_stopwords)

        return all_stopwords

    def get_pagerank(self, wiki_ids):
        """
        returns the number of page rank for a list of doc ids
        """
        page_rank = self.page_rank
        result = []
        for doc_id in wiki_ids:
            if doc_id in page_rank.keys():
                rank = self.page_rank[doc_id]
            else:
                rank = 0
            result.append(rank)

        return result

    def get_pageviews(self, wiki_ids):
        """
        returns the number of page view for a list of doc ids
        """
        page_views = self.page_views
        result = []
        for doc_id in wiki_ids:
            if doc_id in page_views.keys():
                views = self.page_views[doc_id]
            else:
                views = 0
            result.append(views)

        return result

    @staticmethod
    def get_posting_iter(index):
        words, pls = zip(*index.posting_lists_iter(BUCKET_NAME))
        return words, pls


    def get_posting_list(self, index, word):
        """
        retrieves posting list if exists
        """
        try:
            result = index.read_posting_list(word)

        except:
            result = []
   
        return result

    def get_all_postings(self, index, words):
        """
        retrieves all posting lists for a given set of tokens
        """
        words = np.unique(words)
        pls = dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for word in words:
                pls[word] = executor.submit(self.get_posting_list, index, word).result()
   
        
        return list(pls.keys()), pls   

    def tokenize(self, text, with_stem=False):
        """
        tokenize a query into tokens, removes stopwords and if with_stem=True stems the tokens using Porter Stemmer
        """
        
        list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                          token.group() not in self.stop_words]
        if with_stem == True:
            list_of_tokens = [self.stemmer.stem(token) for token in list_of_tokens]

        return list_of_tokens
    
    def result_w_title(self, doc_ids):
        """
        Wrapper to return a list of doc_ids with the matching title (doc_id, title)
        """
        titles = self.titles
        return [(doc_id, titles.get(doc_id)) for doc_id in doc_ids]
    
    def expand_query(self, tokenized_query, pls, index, min_threshold=100, neighbors=2):
        """
        Expands a list of tokens. the condition for query expansion is that the posting list for the given word is under min_threshold. 
        if so, the function retrieves the posting lists for its nearest neigherbors and returns the updated query and postings.
        """       
        query_expansion = []
        for word, pl in pls.items():
            if len(pl) < min_threshold:
                try:
                    extension = self.word_vectors.most_similar(word, topn=neighbors)
                except:
                    extension = []
                for new_word, _ in extension:
                    if new_word not in tokenized_query:
                        query_expansion.append(new_word)
        query_expansion, expansion_pls = self.get_all_postings(index, query_expansion)
        tokenized_query += query_expansion
        pls.update(expansion_pls)
        return tokenized_query, pls

    def get_relevant_titles(self, query) -> list:
        """
        Retrieves relevant titles using binary search without expansion and stemming
        """
        term_counter = self.binary_search(query, self.title_index)
        return self.result_w_title([i[0] for i in term_counter.most_common()])

    def get_relevant_anchors(self, query) -> list:
        """
        Retrieves relevant titles using binary search without expansion and stemming
        """
        term_counter = self.binary_search(query, self.anchor_index)
        return self.result_w_title([i[0] for i in term_counter.most_common()])
    
    def binary_search(self, query, index, with_stem=False, expand=False) -> Counter:
        """
        tokens, stems and extends query per conditions. 
        For each unique term in the query that exists in doc, the doc gets a point.
        Returns a Counter object with {doc_id:count}
        """
        tokenized_query = self.tokenize(query, with_stem)

        tokenized_query, pls = self.get_all_postings(index, tokenized_query)

        if expand:
            tokenized_query, pls = self.expand_query(tokenized_query, pls, index) 
            

        term_counter = Counter()
        for term in np.unique(tokenized_query):
            if term in pls.keys():
                for doc_id, tf in pls[term]:
                    term_counter[doc_id] += 1

        return term_counter

    def tfidf_search(self,query, index, DL, n_results=100, with_stem=False, expand=False):
        """
        tokens, stems and extends query per conditions. 
        Calls tfidf method to find best documents.
        Returns a list of (doc_id, score) sorted by score descending of size 100.
        """
        # tokenize query
        tokenized_query = self.tokenize(query, with_stem)
        tokenized_query = [token for token in tokenized_query if token in index.df.keys()]
        tokenized_query, pls = self.get_all_postings(index, tokenized_query)
        
        if expand:
            tokenized_query, pls = self.expand_query(tokenized_query, pls, index) 
        
        # get top 100 results      
        top_n = tfidf(inverted_index=pls,
                                   df=index.df, 
                                   Q=tokenized_query,
                                   DL=DL,
                                   limit=n_results)

        return top_n


    def bm25_search(self, query, index, DL, n_results=100, with_stem=False, expand=False):
        """
        tokens, stems and extends query per conditions. 
        Calls BM25 method to find best documents.
        Returns a list of (doc_id, score) sorted by score descending of size 100.
        """
        # tokenize query
        tokenized_query = self.tokenize(query, with_stem)
        tokenized_query = [token for token in tokenized_query if token in index.df.keys()]
        tokenized_query, pls = self.get_all_postings(index, tokenized_query)

        if expand:
            tokenized_query, pls = self.expand_query(tokenized_query, pls, index)

        top_n = bm25(inverted_index=pls,
                                   df=index.df, 
                                   Q=tokenized_query,
                                   DL=DL,
                                   avdl= self.avdl,
                                   N = self.N,
                                   limit=n_results)
        

        return top_n

    def get_relevant_bodies(self, query):
        """
        Returns the top 100 results using tfidf cosine similarity of the body text"""
        top_n = self.tfidf_search(query, self.body_index, self.doc_length)
        top_n = [doc_id for doc_id, _ in top_n]
        return self.result_w_title(top_n)

    def all_search_results(self, query, n=100):
        """
        returns a dictionary of the result for a given query of all search methods. used for testing.
        """
        title_DL={doc_id:len(title) for doc_id, title in self.titles.items()}
        text_DL=self.doc_length
        # for testing
        search_scores = dict()
        search_scores['binary_title_weight']=self.binary_search(query, self.title_index).most_common(n)
        search_scores['binary_anchor_weight']=self.binary_search(query, self.anchor_index).most_common(n)
        search_scores['binary_body_weight']=self.binary_search(query, self.body_index).most_common(n)
        search_scores['stem_binary_title_weight']=self.binary_search(query, self.stem_title_index, with_stem=True).most_common(n)
        search_scores['stem_binary_body_weight']=self.binary_search(query, self.stem_body_index, with_stem=True).most_common(n)
        search_scores['tfidf_text_weight']=self.tfidf_search(query, self.body_index, text_DL)
        search_scores['tfidf_title_weight']=self.tfidf_search(query, self.title_index, title_DL)
        search_scores['bm25_title_score']=self.bm25_search(query, self.title_index, title_DL)
        search_scores['bm25_title_stem_score']=self.bm25_search(query, self.stem_title_index, title_DL, with_stem=True)
        search_scores['bm25_text_score']=self.bm25_search(query, self.body_index, text_DL)
        search_scores['bm25_text_stem_score']=self.bm25_search(query, self.stem_body_index, text_DL, with_stem=True)
        return search_scores
    
    @staticmethod 
    def normalize_result(scores):
        """
        Normalizes a list of (id, score) so that max score=1 and returns it
        """
        normalized = []
        if scores:
            max_value = max(scores, key=lambda x: x[1])[1]
            normalized = [(key, value/max_value) for key, value in scores]
        return normalized

    def merge(self, results, weights):
        """
        Merges results by doc_id using the given weights. 
        Adds page rank and page view scores with there weights
        returns a sorted list of the joint score (doc_id, score)
        """
        # Create a dictionary to store the sum of scores for each id
        doc_scores = defaultdict(float)
        for i, result in enumerate(results):
            for id, score in result:
                doc_scores[id] += score * weights[i]

        top_results = dict(sorted(doc_scores.items(), key=itemgetter(1), reverse=True)[:40])

        page_rank = self.page_rank
        page_views = self.page_views
        for doc_id in top_results.keys():
            top_results[doc_id] += weights[-2] * page_rank.get(doc_id, 1) / self.max_pr + weights[-1] * page_views.get(doc_id, 1) / self.max_pv

        return sorted(top_results.items(), key=itemgetter(1), reverse=True)


    def best_search(self, query):
        """
        Uses threading to parallelize retrieval of scores using binary and bm25 methods.
        Merges results.
        Returns the top results as a list of (doc_id, title)
        """
        weights= self.weights
        top_n=self.topn
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            jobs = [executor.submit(self.binary_search, query, self.stem_title_index, with_stem=True, expand=False),
                    executor.submit(self.bm25_search, query, self.stem_body_index, 
                                    self.doc_length, with_stem=True, expand=True)]

            results = [job.result() for job in jobs]
            results[0] =results[0].most_common(20)
            results = [self.normalize_result(res[:20]) for res in results]
        # merge searchs
        results = self.merge(results, weights)
        # return top n results
        return self.result_w_title([doc_id for doc_id, score in results[:top_n]])
   


