__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from pypdf import PdfReader
from docx import Document
from crewai import Crew, Agent, Task
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login
import re

# Autenticar no Hugging Face
try:
    login()
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
    # Remover caracteres especiais e stopwords simples
    text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação

    # Dividir textos longos em pedaços menores
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Tamanho máximo de cada pedaço
        chunk_overlap=200  # Sobreposição entre pedaços
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

# Carregar o modelo LLaMA 2
def load_llama2_model():
    try:
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # Modelo LLaMA 2
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo LLaMA 2: {str(e)}")
        return None

# Função para gerar respostas com o LLaMA 2
def ask_llama2(query, context, llm_pipeline):
    try:
        input_text = f"Contexto: {context}\n\nPergunta: {query}"
        response = llm_pipeline(input_text, max_length=200, do_sample=True, temperature=0.7)
        return response[0]['generated_text']
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {str(e)}")
        return "Erro ao gerar resposta."

# Interface Gráfica com Streamlit
def main():
    st.title("Multi PDF RAG Chatbot com CrewAI e LLaMA 2")
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
                # Carregar o modelo LLaMA 2
                llm_pipeline = load_llama2_model()
                if llm_pipeline:
                    # Criar agentes do CrewAI
                    pdf_search_agent = Agent(
                        role="PDF Search Specialist",
                        goal="Buscar e extrair informações relevantes de documentos PDF.",
                        backstory="Você é um especialista em buscar informações específicas em documentos PDF.",
                        allow_delegation=False,
                        verbose=True
                    )

                    pdf_summarizer_agent = Agent(
                        role="PDF Summarizer",
                        goal="Gerar resumos claros e concisos de textos extraídos de PDFs usando LLaMA 2.",
                        backstory="Você é um especialista em processamento de linguagem natural e geração de resumos.",
                        allow_delegation=False,
                        verbose=True
                    )

                    # Loop do chatbot
                    query = st.text_input("Faça uma pergunta:")
                    if query:
                        # Recuperar documentos relevantes
                        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                        relevant_docs = retriever.get_relevant_documents(query)
                        context = "\n".join([doc.page_content for doc in relevant_docs])

                        # Criar tarefas do CrewAI
                        search_task = Task(
                            description=f"Buscar no PDF pela consulta: '{query}'",
                            agent=pdf_search_agent,
                            expected_output="Informações relevantes extraídas do PDF."
                        )

                        summarize_task = Task(
                            description="Resumir o texto extraído do PDF usando LLaMA 2.",
                            agent=pdf_summarizer_agent,
                            expected_output="Resumo claro e conciso do texto gerado pelo LLaMA 2."
                        )

                        # Criar a equipe (Crew)
                        crew = Crew(
                            agents=[pdf_search_agent, pdf_summarizer_agent],
                            tasks=[search_task, summarize_task],
                            verbose=2  # Mostra logs detalhados do processo
                        )

                        # Executar a equipe
                        result = crew.kickoff(inputs={"query": query, "context": context})

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