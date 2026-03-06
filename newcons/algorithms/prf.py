import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def algo_pseudo_relevance_feedback(query, initial_docs, top_k_keywords=3):
    """PRF: 提取关键词扩展查询"""
    if not initial_docs: return [query]
    doc_texts =[d.page_content for d in initial_docs]
    vectorizer = TfidfVectorizer(max_features=1000, stop_words=['的', '了', '是', '我', '你'])
    try:
        tfidf_matrix = vectorizer.fit_transform(doc_texts)
        feature_names = vectorizer.get_feature_names_out()
        avg_scores = np.array(tfidf_matrix.todense()).mean(axis=0).tolist()[0]
        top_indices = np.argsort(avg_scores)[::-1][:top_k_keywords]
        keywords = [feature_names[i] for i in top_indices]
        return[query, query + " " + " ".join(keywords)]
    except:
        return [query]
        