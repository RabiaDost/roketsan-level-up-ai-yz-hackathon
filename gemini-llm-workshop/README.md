1. LLM Pratikleri
    1.Projenin Oluşturulması ve Sanal ortam
        python -m venv venv
        .\venv\Scripts\activate
    2.Kütüphanelerin kurulması
        langchain
        langchain-google-genai
        pip install -r reuirements.txt
    3 gemini api key:
        .env: GOOGLE_API_KEY=""
    4. ilk gemini çağrısı yapılması
2. RAG
    1. Proje yapısı
        mil_std_document.py
        create_vector_db.py
        rag.py
    2.kütüphane kur
        langchain
        langchain-google-genai
        langchain-community
        langchain-text-splitters
        faiss-cpu
        python-dotenv 
        pip install -r requirements.txt 
    3.mil-std doküman oluşturma 
        https://github.com/turkiyeyapayzekaakademisi/llm-rag-memory-ai-agents

3.Agent