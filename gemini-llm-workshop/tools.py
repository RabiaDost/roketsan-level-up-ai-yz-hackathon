import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag import retriever,chain as rag_chain

load_dotenv()
general_model=model=ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")
general_promt= ChatPromptTemplate.from_messages([("system", "Sen genel sorulalrı cevaplayan bir yapay zeka ajanısın."),
                                                 ("human", "{question}")])
general_chain= general_promt | general_model | StrOutputParser()

@tool
def general_question_tool(question:str) -> str:
    """
    Genel bilgi, günlük yaşam,yazılım,teknoloji,yemek, eğitim ve MIL-STD dokumanıyla ilgili olmayan
    diğer soruları cevaplamak için kullanılır
    """
    response = general_chain.invoke({"question": question})
    return response

@tool
def mil_std_rag_tool(question: str) -> str:
    """
    RKT-MIL-STD-001 teknik dokümanıyla ilgili soruları cevaplamak için kullanılır.
    Çalışma sıcaklığı, depolama sıcaklığı, nem, titreşim, meknaik şok, rakım , güç gereksinimleri ve test kriterleri gibi
    teknik doküman sorularında bu tool kullanılmalıdır.
    """
    documents = retriever.invoke(question)
    context= "\n\n".join(document.page_content for document in documents)
    response = rag_chain.invoke({"context":context, "question":question})
    return response