import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DEFAULT_PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_announcement.pdf")
DEFAULT_INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "faiss_index")
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"

def extract_and_index(pdf_path: str = DEFAULT_PDF_PATH, index_dir: str = DEFAULT_INDEX_DIR):
    print(f"[*] Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[*] Successfully loaded {len(documents)} pages from PDF.")

    # 공문서 텍스트 분할 (KURE-v1의 512 토큰 제한을 고려하여 적절한 청크 크기 설정)
    # 표나 목록 구조가 포함되므로 줄바꿈 기준으로 우선 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"[*] Split into {len(chunks)} text chunks.")

    # KURE-v1 임베딩 모델 로드
    print(f"[*] Initializing Embedding Model ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # FAISS 벡터 스토어 생성
    print("[*] Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 인덱스 로컬 저장
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"[+] FAISS index saved successfully to: {index_dir}")
    return vectorstore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Ingestion and FAISS Vectorstore Indexing")
    parser.add_argument("--pdf", type=str, default=DEFAULT_PDF_PATH, help="Path to PDF file")
    parser.add_argument("--out", type=str, default=DEFAULT_INDEX_DIR, help="Path to save FAISS index")
    args = parser.parse_args()

    extract_and_index(args.pdf, args.out)
