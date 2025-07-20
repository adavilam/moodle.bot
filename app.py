# --- 1. Importaciones ---
import streamlit as st
import os
from dotenv import load_dotenv

# Importaciones de LangChain para el Agente ReAct
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent

# Parche para el bucle de eventos asíncronos en Streamlit
import nest_asyncio
nest_asyncio.apply()

# --- 2. Configuración y Carga de API Keys ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not google_api_key or not tavily_api_key:
    st.error("Asegúrate de tener definidas GOOGLE_API_KEY y TAVILY_API_KEY en tu archivo .env")
    st.stop()

# --- 3. Función para crear las herramientas del Agente (con caché) ---
@st.cache_resource
def cargar_herramientas():
    loader = TextLoader("conocimiento.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documentos_divididos = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectorstore = FAISS.from_documents(documentos_divididos, embeddings)
    retriever = vectorstore.as_retriever()

    # HERRAMIENTA 1: Búsqueda en los documentos del curso
    herramienta_documentos = create_retriever_tool(
        retriever,
        "busqueda_documentos_curso",
        "Busca en los documentos del curso. Úsala para preguntas sobre tareas, fechas, contenido específico o el sílabo del curso."
    )

    # HERRAMIENTA 2: Búsqueda en Internet
    herramienta_web = TavilySearchResults(max_results=2, description="Busca en internet información general o para profundizar en un tema.")
    
    return [herramienta_documentos, herramienta_web]

# --- 4. Creación del Agente (Método Alternativo 'ReAct') ---
herramientas = cargar_herramientas()
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0)

# Este es el prompt específico para el agente tipo 'ReAct'
template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt_agente = PromptTemplate.from_template(template)

# Usamos 'create_react_agent', que es compatible con versiones más antiguas
agente = create_react_agent(llm, herramientas, prompt_agente)
agent_executor = AgentExecutor(agent=agente, tools=herramientas, verbose=True, handle_parsing_errors=True)

# --- 5. Interfaz de Streamlit ---
st.title("🤖 Asistente Investigador del Curso (Plan B)")
st.caption("Puedo buscar en los documentos del curso y también en internet.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_usuario := st.chat_input("¿Qué quieres saber?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Investigando... 🕵️‍♂️"):
            # 'agent_scratchpad' se maneja internamente por el AgentExecutor
            respuesta = agent_executor.invoke({
                "input": prompt_usuario,
            })
            st.markdown(respuesta["output"])

    st.session_state.chat_history.append({"role": "assistant", "content": respuesta["output"]})