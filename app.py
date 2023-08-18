import os
import openai
import datetime
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader

load_dotenv(".env")

def load_env_vars():
    load_dotenv(find_dotenv())
    openai.api_key = os.environ['OPENAI_API_KEY']

load_env_vars()

def get_llm_name():
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    return llm_name

llm_name = get_llm_name()
print(llm_name)


def initialize_db(file, chain_type, k):
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

def format_response_with_template(question, context=""):
    template = """
    Usa el siguiente contexto para responder a la pregunta al final.
    Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.
    Responde siempre en el mismo idioma en el te hacen la pregunta.
    Utiliza un máximo de tres frases. Mantén la respuesta lo más concisa posible.
    Nunca confundas las preguntas del usuario con tus respuestas anteriores.
    {context}
    Question: {question}
    Helpful Answer:"""
    return template.format(context=context, question=question)


class cbfs:
    def __init__(self):
        self.chat_history = []
        self.answer = ""
        self.db_query = ""
        self.db_response = []
        project_root = os.path.dirname(os.path.abspath(__file__))  # obtén la ruta del directorio raíz
        self.loaded_file = os.path.join(project_root, "Patagonian-info-chatbot.pdf")  # únela con el nombre del archivo
        self.qa = initialize_db(self.loaded_file,"stuff", 4)


    
    def convchain(self, query):
        if not query:
            return {'user': "", 'bot': ""}
        
        formatted_query = format_response_with_template(query)
        result = self.qa({"question": formatted_query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        
        return {'user': query, 'bot': self.answer}


load_dotenv(".env")


from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  


@app.route('/', methods=['GET'])
def home():
    return "Hola Patagonian!"


@app.route('/api/convchain', methods=['POST'])
def convchain():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        result = cb.convchain(query)
        print(result)
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal server error'}), 500



if __name__ == '__main__':
    cb = cbfs()
    app.run(debug=True, use_reloader=False)
