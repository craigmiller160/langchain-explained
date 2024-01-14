from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pipetools import pipe
import os.path

OLLAMA_HOST = 'http://192.168.7.232:11434'
OLLAMA_MODEL = 'llama2'
CHROMA_DB_DIR = './chromadb'
DOC_COUNT_FILE = f'{CHROMA_DB_DIR}/doc_count.txt'


ollama = Ollama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
oembed = OllamaEmbeddings(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
chroma = Chroma(persist_directory='./chromadb', embedding_function=oembed)


def simple_prompt():
    result = ollama('Why is the sky blue?')
    print(result)


string_length = pipe | len | str


def download_and_split_text():
    loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    return text_splitter.split_documents(data)


def add_documents_to_chroma(docs, num_to_add):
    existing_count = 0
    if os.path.exists(DOC_COUNT_FILE):
        with open(DOC_COUNT_FILE) as file:
            existing_count = file.read() > pipe | int

    docs_to_add = docs[existing_count:existing_count + num_to_add]
    chroma.add_documents(docs_to_add)

    with open(DOC_COUNT_FILE, 'w') as file:
        file.write(str(existing_count + num_to_add))


def document_prompt():
    print("Loading gutenberg text")
    loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
    data = loader.load()

    print("Splitting text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    print("Number of splits: " + string_length(all_splits))

    print("Setting up vector store")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

    print("Doing vectorstore similarity search")
    question = "Who is Neleus and who is in Neleus' family?"
    docs = vectorstore.similarity_search(question)
    print("Number of Docs: " + string_length(docs))

    print("Using vectorstore for llm query")
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    result = qachain({"query": question})
    print(result)


docs = download_and_split_text()
add_documents_to_chroma(docs, 1)
