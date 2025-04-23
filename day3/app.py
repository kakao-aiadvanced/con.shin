from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from tavily import TavilyClient
from langchain_openai import ChatOpenAI


tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
llm = ChatOpenAI(model='gpt-4o-mini')
vectorstore = None

def _set_up():
    ## retriever
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    global vectorstore
    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    
"""
exmaple: retrieval_grader.invoke({"question": question, "document": doc_txt})
-> {"score": "yes" or "no"}
"""

_relevance_checker = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are a grader assessing relevance
            of a retrieved document to a user question. If the document contains keywords related to the user question,
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """),
        ("human", "question: {question}\n\n document: {document} "),
    ]
) | llm | JsonOutputParser()

_answer_generator = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise"""
        ),
        ("human", "question: {question}\n\n context: {context} "),
    ]
) | llm | StrOutputParser()

_hallucination_checker = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are a grader assessing whether
            an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
            single key 'score' and no preamble or explanation."""),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
) | llm | JsonOutputParser()

from langchain_core.documents import Document

### nodes

def docs_retrieval(state):
    question = state["question"]
    documents = vectorstore.as_retriever().invoke(question)
    return {"documents": documents, "question": question}


def relevance_checker(state):
    question = state["question"]
    documents = state["documents"]
    doc_txt = "\n\n".join([doc.page_content for doc in documents])
    result = _relevance_checker.invoke({"question": question, "document": doc_txt})
    return {"documents": documents, "question": question, "relevance": result["score"]}

def generator_answer(state):
    question = state["question"]
    documents = state["documents"]
    doc_txt = "\n\n".join([doc.page_content for doc in documents])
    result = _answer_generator.invoke({"question": question, "context": doc_txt})
    return {"documents": documents, "question": question, "generation": result}

def hallucination_checker(state):
    generation = state["generation"]
    documents = state["documents"]
    try_count = state.get("try_count", 0)
    result = _hallucination_checker.invoke({"documents": documents, "generation": generation})
    
    if "try_count" in state:
        state["try_count"] += 1
    else:
        state["try_count"] = 1
    
    if try_count > 2:
        hallucination = "failed"
    else:
        hallucination = result["score"]
    
    return {
        "documents": documents, 
        "generation": generation, 
        "hallucination": hallucination, 
        "try_count": try_count}
    
def web_search(state):
    question = state["question"]
    docs = tavily.search(query=question)['results']
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": [web_results], "question": question}

### State
from langgraph.graph import END, StateGraph
from typing import List, Literal, TypedDict


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    relevance: Literal["yes", "no"]
    hallucination: Literal["yes", "no"]
    try_count: int

def _get_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("docs_retrieval", docs_retrieval)
    workflow.add_node("relevance_checker", relevance_checker)
    workflow.add_node("generator_answer", generator_answer) 
    workflow.add_node("hallucination_checker", hallucination_checker)
    workflow.add_node("web_search", web_search)

    workflow.set_entry_point("docs_retrieval")
    workflow.add_edge("docs_retrieval", "relevance_checker")
    workflow.add_conditional_edges(
        "relevance_checker",
        lambda state: state["relevance"],
        {
            "yes": "generator_answer",
            "no": "web_search",
        },
    )
    workflow.add_edge("web_search", "relevance_checker")
    workflow.add_edge("generator_answer", "hallucination_checker")
    workflow.add_conditional_edges(
        "hallucination_checker",
        lambda state: state["hallucination"],
        {
            "yes": END,
            "no": "generator_answer",
            "failed": END,
        },
    )

    return workflow

def get_app():
    _set_up()
    workflow = _get_workflow()
    app = workflow.compile()
    return app
