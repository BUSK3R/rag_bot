import os
from typing import List, Tuple, Any, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGBot:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.db_dir = os.path.join(self.data_dir, "faiss_index")
        self.embed_model_name = "nlpai-lab/KURE-v1"
        self.device = "cpu"
        
        self.embeddings = None
        self.vectorstore = None

    def load_vectorstore(self):
        print(f"[*] Loading embedding model ({self.embed_model_name})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if not os.path.exists(self.db_dir):
            raise FileNotFoundError(f"FAISS index not found at {self.db_dir}. Please run ingest.py first.")
            
        print(f"[*] Loading FAISS index from {self.db_dir}...")
        self.vectorstore = FAISS.load_local(
            self.db_dir, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("[+] FAISS Vectorstore successfully loaded.")

    def load_llm(self):
        """
        Placeholder for LLM loading.
        """
        print("[*] LLM component has been removed per user request.")

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[Any, float]]:
        if not self.vectorstore:
            self.load_vectorstore()
        # similarity_search_with_score returns (doc, distance)
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def generate_answer(self, query: str, k: int = 3) -> Dict[str, Any]:
        if not self.vectorstore:
            self.load_vectorstore()
            
        print(f"[*] Retrieval phase for query: '{query}'")
        retrieved_docs = self.retrieve(query, k=k)
        
        context_texts = []
        for doc, score in retrieved_docs:
            context_texts.append(doc.page_content)
        
        context = "\n\n".join(context_texts)
        prompt = f"문맥:\n{context}\n\n질문: {query}\n\n답변:"

        print("[*] LLM generation phase skipped.")
        return {
            "query": query,
            "answer": "LLM removed. RAG Pipeline framework only.",
            "source_documents": retrieved_docs
        }
