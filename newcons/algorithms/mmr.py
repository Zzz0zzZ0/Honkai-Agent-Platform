import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def algo_mmr_rerank(query, docs, embeddings, k_param=3, lambda_mult=0.5):
    """MMR: 多样性重排序 (受 k_param 控制)"""
    if not docs: return[]
    final_k = min(k_param, len(docs))
    
    query_embed = np.array([embeddings.embed_query(query)])
    doc_embeds = np.array(embeddings.embed_documents([d.page_content for d in docs]))
    
    selected_indices =[]
    candidate_indices = list(range(len(docs)))
    
    for _ in range(final_k):
        best_score = -np.inf
        best_idx = -1
        for idx in candidate_indices:
            sim_q = cosine_similarity([doc_embeds[idx]], query_embed)[0][0]
            redundancy = 0
            if selected_indices:
                redundancy = np.max(cosine_similarity([doc_embeds[idx]], doc_embeds[selected_indices]))
            mmr_score = (1 - lambda_mult) * sim_q - (lambda_mult * redundancy)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
    
    return [docs[i] for i in selected_indices]