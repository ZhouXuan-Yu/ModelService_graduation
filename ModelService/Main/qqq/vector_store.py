import chromadb
import os
from chromadb.config import Settings
import PyPDF2
import docx
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re

class LocalEmbeddingFunction:
    def __init__(self):
        # 添加自定义词典（如果有的话）
        # jieba.load_userdict("path/to/your/dict.txt")
        
        # 将停用词转换为列表
        self.stop_words = ['的', '了', '和', '是', '就', '都', '而', '及', '与', '着']
        
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            max_features=1536,  # 设置向量维度
            stop_words=self.stop_words,  # 使用列表形式的停用词
            ngram_range=(1, 2)  # 支持bigram特征
        )
        self.fitted = False
    
    def _tokenize(self, text):
        """分词预处理"""
        # 清理文本
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = jieba.cut(text, cut_all=False)
        # 过滤空字符和停用词
        return [w for w in words if w.strip() and w not in self.stop_words]
    
    def __call__(self, input):
        if not isinstance(input, list):
            input = [input]
            
        # 确保所有文本都是字符串
        texts = [str(t) for t in input]
        
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        
        try:
            vectors = self.vectorizer.transform(texts).toarray()
        except ValueError:
            # 如果遇到新词导致的错误，重新训练vectorizer
            self.vectorizer.fit(texts)
            vectors = self.vectorizer.transform(texts).toarray()
        
        # 标准化向量
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        
        # 如果向量维度不足1536，用0填充
        if vectors.shape[1] < 1536:
            padding = np.zeros((vectors.shape[0], 1536 - vectors.shape[1]))
            vectors = np.hstack([vectors, padding])
            
        return vectors.tolist()

class VectorStore:
    def __init__(self):
        # 设置固定存储目录
        self.storage_dir = r"E:\MyUser\GraduationDesign\Programe\FixedStorage"
        self.data_dir = os.path.join(self.storage_dir, 'vector_store')
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化嵌入函数
        self.embedding_function = LocalEmbeddingFunction()
        
        # 初始化 ChromaDB，设置持久化存储
        self.client = chromadb.Client(Settings(
            persist_directory=os.path.join(self.data_dir, 'chroma'),
            is_persistent=True
        ))
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_function
        )
        
        # JSON文件路径
        self.json_path = os.path.join(self.data_dir, 'knowledge_base.json')
        self.load_json()

    def load_json(self):
        """从JSON文件加载数据"""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 如果数据库为空，则导入JSON数据
                if not self.get_all_documents():
                    for item in data:
                        self.add_documents([item['content']], metadata=[item.get('metadata', {})])

    def save_json(self):
        """保存数据到JSON文件"""
        documents = self.get_all_documents()
        metadata = self.collection.get()['metadatas']
        data = [{'content': doc, 'metadata': meta} for doc, meta in zip(documents, metadata)]
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def extract_text_from_pdf(self, file_path):
        """从PDF文件提取文本"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"PDF处理错误: {e}")
            return ""

    def extract_text_from_docx(self, file_path):
        """从DOCX文件提取文本"""
        try:
            # 对于 .doc 文件，尝试用 docx 打开可能会失败
            if file_path.lower().endswith('.doc'):
                print("不支持旧版 .doc 格式，请转换为 .docx 格式")
                return ""
                
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Word文档处理错误: {e}")
            return ""

    def add_documents(self, texts, ids=None, metadata=None):
        if ids is None:
            # 获取当前文档数量作为起始ID
            current_count = len(self.get_all_documents())
            ids = [str(i) for i in range(current_count, current_count + len(texts))]
        if metadata is None:
            metadata = [{"source": "default"} for _ in texts]
        
        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadata
        )
        self.save_json()  # 保存到JSON文件

    def add_file(self, file_path):
        """添加文件（支持PDF和DOCX）"""
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            text = ""
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                text = self.extract_text_from_docx(file_path)
            elif file_ext == '.doc':
                raise ValueError("不支持旧版 .doc 格式，请转换为 .docx 格式")
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
            
            if text.strip():
                self.add_documents(
                    [text],
                    metadata=[{"source": os.path.basename(file_path)}]
                )
                return True
            return False
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return False

    def query(self, query_text, n_results=3):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    def get_all_documents(self):
        """获取所有文档"""
        results = self.collection.get()
        return results['documents'] if results else []

    def delete_document(self, index):
        """删除指定索引的文档"""
        try:
            all_docs = self.get_all_documents()
            if 0 <= index < len(all_docs):
                doc_id = str(index)
                self.collection.delete(ids=[doc_id])
                self.save_json()  # 保存到JSON文件
                return True
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False 