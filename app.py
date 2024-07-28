
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
import huggingface_hub
from llama_index.core import PromptTemplate
from flask import render_template, request,jsonify
import torch
import os

# Initialize FastAPI application
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Login to Hugging Face Hub using your API token
huggingface_hub.login(token="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")

UPLOAD_FOLDER = 'uploads_new'
ALLOWED_EXTENSIONS = {'pdf'}

# Define upload folder and allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define the system prompt for the LLM
system_prompt = """
    You are an AI assistant designed to help users interact with the contents of a PDF document. Your role is to understand the context of the document and provide relevant information based on the user's queries. You can extract key terms, phrases, and sentences from the document to generate accurate and helpful responses.
    
    Guidelines:
    1. Analyze the user's question and determine the relevant section of the PDF document.
    2. Provide concise and informative answers based on the extracted content from the document.
    3. If the information is not found in the document, inform the user that the specific information is not available.
    4. Avoid making assumptions or providing information that is not present in the document.
    5. Be respectful and polite in all interactions.
    
    Example:
    Context: [Extracted content from the PDF document]
    Question: "What are the key points of the introduction section?"
    Answer: "The introduction section highlights the main objectives of the document, provides an overview of the topics covered, and sets the stage for the detailed discussions in the subsequent sections."
    
    If greeted, respond accordingly, and if bid farewell, respond with "goodbye".
Context:\n {context}?\n
Question: \n{question}\n
"""


# Define the query wrapper prompt
query_wrapper_prompt = PromptTemplate("{query_str}")

# Initialize the LLM for PDF interaction
llm2 = HuggingFaceInferenceAPI(
    context_window=8192,
    max_new_tokens=1000,
    generate_kwargs={"temperature": 0.3, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Initialize the embedding model
embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Set global settings for the application
Settings.llm = llm2
Settings.embed_model = embed_model

@app.route('/')
def index():
    return render_template('index.html')
@app.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext in {'.pdf'}:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        return JSONResponse(content={"success": True, "message": "File uploaded successfully"})
    else:
        return JSONResponse(content={"success": False, "message": "Unsupported file format"})

@app.post("/ask_pdf")
async def ask_pdf(question: str = Form(...)):
    documents = SimpleDirectoryReader('uploads_new').load_data()
    Settings.llm = llm2
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    print(response)
    return JSONResponse(content={"response": str(response)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
