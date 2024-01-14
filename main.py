from langchain_community.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

OLLAMA_HOST = 'http://192.168.7.232:11434'
OLLAMA_MODEL = 'llama2'

ollama = Ollama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
oembed = OllamaEmbeddings(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)


def simple_prompt():
    result = ollama('Why is the sky blue?')
    print(result)


def document_prompt():
    print("Loading gutenberg text")
    loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
    data = loader.load()

    print("Splitting text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    print("Number of splits: " + str(len(all_splits)))

    print("Setting up vector store")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

    print("Doing vectorstore similarity search")
    question = "Who is Neleus and who is in Neleus' family?"
    docs = vectorstore.similarity_search(question)
    print("Number of Docs: " + str(len(docs)))

    print("Using vectorstore for llm query")
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    result = qachain({"query": question})
    print(result)


document_prompt()
