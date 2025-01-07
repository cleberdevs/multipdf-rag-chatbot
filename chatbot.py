__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from pypdf import PdfReader
from docx import Document
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import re
import torch

# Verificar se o PyTorch está funcionando corretamente
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    st.error(f"Erro ao verificar o PyTorch: {str(e)}")

# Autenticar no Hugging Face
try:
    login(token="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")
    st.success("Autenticado no Hugging Face com sucesso!")
except Exception as e:
    st.error(f"Erro ao autenticar no Hugging Face: {str(e)}")

# Função para extrair texto de um PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Erro ao ler o PDF {pdf_path}: {str(e)}")
        return ""

# Função para extrair texto de um DOCX
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Erro ao ler o DOCX {docx_path}: {str(e)}")
        return ""

# Função para extrair texto de um TXT
def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
        return text
    except Exception as e:
        st.error(f"Erro ao ler o TXT {txt_path}: {str(e)}")
        return ""

# Função para carregar múltiplos documentos e extrair textos
def load_documents_from_folder(folder_path):
    texts = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_name.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file_name.endswith(".txt"):
            text = extract_text_from_txt(file_path)
        else:
            st.warning(f"Formato não suportado: {file_name}")
            continue
        if text:
            texts.append(text)
    return texts

# Pré-processamento de texto (remoção de stopwords e divisão de textos longos)
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Tamanho máximo de cada pedaço
        chunk_overlap=100  # Sobreposição entre pedaços
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Configuração do modelo de embeddings (usando Sentence Transformers)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configuração do banco de dados vetorial (Chroma)
def create_vector_store(texts):
    try:
        vector_store = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
        return vector_store
    except Exception as e:
        st.error(f"Erro ao criar o banco de dados vetorial: {str(e)}")
        return None

# Função para carregar o modelo GPT-Neo 2.7B (sem CUDA)
def load_model():
    try:
        model_name = "EleutherAI/gpt-neo-2.7B"  # Modelo GPT-Neo 2.7B
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)  # Carregar na CPU
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,  # Aumente o valor conforme necessário
            max_new_tokens=200,  # Define o número máximo de tokens na resposta
            temperature=0.7,  # Controla a criatividade da resposta
            do_sample=True  # Permite amostragem para respostas mais variadas
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

# Função para criar um agente de busca
def create_search_agent(vector_store, llm):
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})  # Aumente o número de documentos
        )
        return qa_chain
    except Exception as e:
        st.error(f"Erro ao criar o agente de busca: {str(e)}")
        return None

# Interface Gráfica com Streamlit
def main():
    st.title("Multi PDF RAG Chatbot com LangChain e GPT-Neo 2.7B")
    st.write("Faça perguntas com base em documentos PDF, DOCX ou TXT.")

    # Inicializar o histórico de conversas
    if "history" not in st.session_state:
        st.session_state.history = []

    # Upload de arquivos
    uploaded_files = st.file_uploader("Carregue seus documentos (PDF, DOCX, TXT)", accept_multiple_files=True)
    if uploaded_files:
        folder_path = "./uploaded_files"
        os.makedirs(folder_path, exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join(folder_path, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Arquivos carregados com sucesso!")

        # Carregar documentos
        texts = load_documents_from_folder(folder_path)
        if texts:
            # Pré-processar textos
            processed_texts = []
            for text in texts:
                processed_texts.extend(preprocess_text(text))

            # Criar banco de dados vetorial
            vector_store = create_vector_store(processed_texts)
            if vector_store:
                # Carregar o modelo
                llm = load_model()
                if llm:
                    # Criar agente de busca
                    search_agent = create_search_agent(vector_store, llm)

                    # Loop do chatbot
                    query = st.text_input("Faça uma pergunta:")
                    if query:
                        # Executar a busca
                        result = search_agent.run(query)

                        # Adicionar ao histórico
                        st.session_state.history.append({"Pergunta": query, "Resposta": result})

                    # Exibir o histórico de conversas
                    st.write("### Histórico de Conversas")
                    for entry in st.session_state.history:
                        st.write(f"**Pergunta:** {entry['Pergunta']}")
                        st.write(f"**Resposta:** {entry['Resposta']}")
                        st.write("---")

if __name__ == "__main__":
    main()

