import gradio as gr
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
HF_LLM_API_URL = os.environ.get('HF_LLM_API_URL')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize chat history and vector database
chat_history = []
vector_db = None

# Function to load and process documents
def load_and_process_docs(files):
    global vector_db
    docs_path = "./temp_docs"
    os.makedirs(docs_path, exist_ok=True)
    for file in files:
        with open(os.path.join(docs_path, file.name), "wb") as f:
            f.write(file.getbuffer())
    loader = PyPDFDirectoryLoader(docs_path, glob='*.pdf')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/stsb-xlm-r-multilingual",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    index = faiss.IndexFlatL2(len(embeddings.embed_query("legal agreement")))
    vector_db = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})
    vector_db.add_documents(chunks)
    return "Documents processed successfully!"

# Function to generate bot response
def generate_response(user_query):
    global chat_history, vector_db
    if vector_db:
        retriever = vector_db.as_retriever()
        llm = OpenAI(openai_api_key=OPENAI_API_KEY)  
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
        result = qa({"question": user_query, "chat_history": chat_history})
        chat_history.append((user_query, result["answer"]))
        return result["answer"]
    else:
        return "Please upload documents first."

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Legal Agreement RAG Bot")
    with gr.Row():
        with gr.Column():
            file_output = gr.File(file_count="multiple", label="Upload Legal Agreements (PDFs)")
            upload_button = gr.Button("Process Documents")
            upload_button.click(load_and_process_docs, inputs=file_output, outputs=gr.Textbox())
        with gr.Column():
            chatbot = gr.Chatbot()
            message = gr.Textbox(label="Enter your query")
            message.submit(generate_response, inputs=message, outputs=chatbot)

demo.launch()