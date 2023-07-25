# chatbot.py
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
import openai
import sys
import datetime
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader


load_dotenv(".env")

def load_env_vars():
    _ = load_dotenv(find_dotenv())
    openai.api_key  = os.environ['OPENAI_API_KEY']

def get_llm_name():
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    return llm_name

llm_name = get_llm_name()
print(llm_name)

embedding = OpenAIEmbeddings()
vectordb = Chroma(embedding_function=embedding)

question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)

llm = ChatOpenAI(model_name=llm_name, temperature=0)

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

def load_db(file, chain_type, k):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

class cbfs:
    def __init__(self):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        self.loaded_file = r"C:\Users\joanm\Downloads\Patagonian data.pdf"
        self.qa = load_db(self.loaded_file,"stuff", 4)
    
    def convchain(self, query):
        if not query:
            return {'user': "", 'bot': ""}

        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 

        return {'user': query, 'bot': self.answer}


load_dotenv(".env")

cb = cbfs()  # Crear una instancia de tu clase cbfs

app = Flask(__name__)

@app.route('/api/convchain', methods=['POST'])
def convchain():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    result = cb.convchain(query)
    print(result)
    return jsonify(result)

@app.route('/api/load_db', methods=['POST'])
def load_db():
    data = request.get_json()
    file = data.get('file')
    chain_type = data.get('chain_type')
    k = data.get('k')
    if not all([file, chain_type, k]):
        return jsonify({'error': 'Missing parameters'}), 400
    result = cb.load_db(file, chain_type, k)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
