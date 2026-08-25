from rag_engine import RAGBot

def test():
    print("[*] 테스트 질문 1: 청년창업사관학교 지원 대상과 나이 조건은 어떻게 되나요?")
    rag = RAGBot()
    res1 = rag.generate_answer("청년창업사관학교 지원 대상과 나이 조건은 어떻게 되나요?")
    print(f"\n[A] 답변:\n{res1['answer']}")
    print("\n[참고 문서]")
    for i, (doc, score) in enumerate(res1['source_documents']):
        print(f"  {i+1}. 페이지 {doc.metadata.get('page', '?')}, 유사도: {score:.4f}")
        
    print("\n" + "="*50 + "\n")
    print("[*] 테스트 질문 2: 지원 혜택(정부지원금 등)은 어느정도인가요?")
    res2 = rag.generate_answer("지원 혜택(정부지원금 등)은 어느정도인가요?")
    print(f"\n[A] 답변:\n{res2['answer']}")
    print("\n[참고 문서]")
    for i, (doc, score) in enumerate(res2['source_documents']):
        print(f"  {i+1}. 페이지 {doc.metadata.get('page', '?')}, 유사도: {score:.4f}")

if __name__ == "__main__":
    test()
