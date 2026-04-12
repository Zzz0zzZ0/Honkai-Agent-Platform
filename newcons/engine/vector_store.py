import os
import shutil
import pandas as pd
from sklearn.decomposition import PCA
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import EMBED_MODEL_NAME

print("🗂️ [Engine] 正在激活海马体 (Embedding Models)...")

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def build_hybrid_knowledge_base(file_path: str):
    """构建 Chroma + BM25 混合索引"""
    persist_dir = "./chroma_db_data"
    # 移除自动删除旧目录的代码，以允许增量叠加知识库

    if file_path.endswith('.pdf'): 
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    else: 
        # 尝试多种编码方式加载文本文件
        docs = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                docs = loader.load()
                break
            except (UnicodeDecodeError, Exception) as e:
                if encoding == 'latin-1':  # latin-1是最后的fallback
                    raise ValueError(f"无法使用任何编码读取文件 {file_path}: {str(e)}")
                continue

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
    # 不再在此处创建局部的 BM25，而是返回 splits 供上层构建全局 BM25
    return vectorstore, splits, len(splits)


def visualize_semantic_space(vectorstore):
    """降维可视化语义空间"""
    data = vectorstore.get(include=["embeddings", "documents"])
    if len(data["embeddings"]) < 3:
        return None

    reducer = PCA(n_components=2)
    vecs_2d = reducer.fit_transform(data["embeddings"])
    df = pd.DataFrame(vecs_2d, columns=["x", "y"])
    df["text"] = [t[:50] + "..." for t in data["documents"]]
    return df

