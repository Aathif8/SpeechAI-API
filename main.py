# Importing Libraries
import os
import uvicorn
import openai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import tempfile

# Initializing FastAPI app
app = FastAPI()

# OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
#HuggingFaceHub Key
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Configuration
# MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = hf_hub_download(repo_id="Aathif/mistral-7b-instruct-v0.1.Q4_K_M.gguf", filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf", token=HF_TOKEN)
CHROMA_DB_PATH = tempfile.mkdtemp()


# Load Mistral model
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=4096, n_threads=os.cpu_count(), f16_kv=True, verbose=False)

# Global retriever
retriever = None

# Data Upload API
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    global retriever

    if not file:
        raise HTTPException(status_code=400, detail="No file received")

    # Read file content into memory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".PDF") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # Process The File
    pdf_loader = PyMuPDFLoader(temp_file_path)
    docs = pdf_loader.load()

    # Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # Store in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=CHROMA_DB_PATH)

    # Create retriever
    retriever = vectorstore.as_retriever()

    return {"message": "File Uploaded and Processed Successfully!"}

# Speech-to-Text Function
def Transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
    
    with open(temp_audio_path, "rb") as audio_file:    
        response = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return response.text

# Text-To-Speech Function
def generate_speech(text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    #Save audio to temp file
    temp_audio_path = "output_audio.mp3"
    with open(temp_audio_path, "wb") as audio_file:
        audio_file.write(response.content)
    return temp_audio_path

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    global retriever

    # Save the uploaded file
    audio_bytes = await file.read()

    # Transcription of Audio
    transcribed_text = Transcribe(audio_bytes)

    # Query LLM (RAG Model)
    if retriever is None:
        return {"error": "No document uploaded for response"}

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = rag_chain.invoke({"query": transcribed_text})

    # Extract response text
    response_text = response["result"] if isinstance(response, dict) and "result" in response else str(response)

    # Text to Speech Response
    output_audio = generate_speech(response_text)

    return {
        "transcription": transcribed_text,
        "response": output_audio,
        "audio_file": FileResponse(output_audio, media_type="audio/mpeg", filename="response.mp3")
    }

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)