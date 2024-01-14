from langchain_community.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# TODO maybe need: pip install GPT4All chromadb

OLLAMA_HOST = 'http://localhost:11434'
OLLAMA_MODEL = 'llama2'

ollama = Ollama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
oembed = OllamaEmbeddings(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)


def simple_prompt():
    result = ollama('Why is the sky blue?')
    print(result)


def document_prompt():
    loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

    question = "Who is Neleus and who is in Neleus' family?"
    docs = vectorstore.similarity_search(question)

    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    result = qachain({"query": question})
    print(result)


simple_prompt()
