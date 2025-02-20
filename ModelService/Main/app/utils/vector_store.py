import chromadb
import os
from chromadb.config import Settings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

class LocalEmbeddingFunction:
    def __init__(self):
        self.stop_words = ['的', '了', '和', '是', '就', '都', '而', '及', '与', '着']
        
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            max_features=1536,
            stop_words=self.stop_words,
            ngram_range=(1, 2)
        )
        self.fitted = False
    
    def _tokenize(self, text):
        """分词预处理"""
        text = text.replace('\n', ' ')
        words = jieba.cut(text, cut_all=False)
        return [w for w in words if w.strip() and w not in self.stop_words]
    
    def __call__(self, input):
        if not isinstance(input, list):
            input = [input]
            
        texts = [str(t) for t in input]
        
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        
        try:
            vectors = self.vectorizer.transform(texts).toarray()
        except ValueError:
            self.vectorizer.fit(texts)
            vectors = self.vectorizer.transform(texts).toarray()
        
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        
        if vectors.shape[1] < 1536:
            padding = np.zeros((vectors.shape[0], 1536 - vectors.shape[1]))
            vectors = np.hstack([vectors, padding])
            
        return vectors.tolist()

class VectorStore:
    def __init__(self):
        # 设置固定存储目录
        self.storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'vector_store')
        
        # 确保目录存在
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 初始化嵌入函数
        self.embedding_function = LocalEmbeddingFunction()
        
        # 初始化 ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=os.path.join(self.storage_dir, 'chroma'),
            is_persistent=True
        ))
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="route_history",
            embedding_function=self.embedding_function
        )

    def add_documents(self, texts, ids=None, metadata=None):
        if ids is None:
            current_count = len(self.get_all_documents())
            ids = [str(i) for i in range(current_count, current_count + len(texts))]
        if metadata is None:
            metadata = [{"source": "route"} for _ in texts]
        
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadata
        )

    def query(self, query_text, n_results=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def get_all_documents(self):
        results = self.collection.get()
        return results['documents'] if results else []

    def delete_document(self, index):
        try:
            all_docs = self.get_all_documents()
            if 0 <= index < len(all_docs):
                doc_id = str(index)
                self.collection.delete(ids=[doc_id])
                return True
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False 