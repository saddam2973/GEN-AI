import os
import time
import json
import gradio as gr
from typing import List, Tuple
from datetime import datetime

# LangChain Imports
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# --- Global Styles & JS ---
# Custom CSS for a beautiful, modern look
custom_css = """
/* 1. Remove the main container border, shadow, and background clipping */
.gradio-container {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    max-width: 100% !important; /* Ensure it fills the space */
}

/* 2. Remove the border from the internal block wrapper */
.main, .gradio-app {
    border: none !important;
    box-shadow: none !important;
}

/* 3. Remove the blue outline that appears when you click/type (Focus Ring) */
*:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* --- Your Previous Styles (Kept for consistency) --- */
.chat-window {
    height: 600px;
    overflow-y: auto;
}
.primary-btn {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    color: white !important;
    border: none !important;
}
.dark-mode-btn {
    background: transparent;
    border: 1px solid #ddd;
}
"""

# JavaScript for Dark Mode Toggle
js_func = """
function() {
    document.body.classList.toggle('dark');
}
"""

# --- RAG Logic ---

def process_documents(files, api_key):
    """
    Ingest and index multiple PDF files. 
    This is separated from the chat loop for optimization.
    """
    if not files:
        return None, "⚠️ Please upload files first."
    
    if not api_key:
        return None, "⚠️ Please enter your Mistral API Key in the sidebar."

    os.environ["MISTRAL_API_KEY"] = api_key
    
    all_docs = []
    status_msg = f"Processing {len(files)} files..."
    
    try:
        for file_path in files:
            loader = PyPDFLoader(file_path)
            all_docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=200
        )
        chunks = splitter.split_documents(all_docs)

        if not chunks:
            return None, "❌ No text found in documents."

        embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore, f"✅ Indexed {len(chunks)} chunks from {len(files)} documents!"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def format_history_for_download(history):
    """Convert chat history to a downloadable JSON file."""
    if not history:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
        
    return filename

def chat_engine(message, history, vectorstore, api_key, model_name, temperature):
    """
    Core Chat Logic: Uses 'yield' to show user message immediately.
    """
    # --- STEP 1: Show User Message Immediately ---
    # Append user message to history
    history.append({"role": "user", "content": message})
    # Yield the updated history so it appears on screen NOW
    yield history, history, "" 

    if not api_key:
        response_text = "⚠️ Please enter your Mistral API Key in the settings."
        history.append({"role": "assistant", "content": response_text})
        yield history, history, ""
        return

    # Initialize LLM
    try:
        llm = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            mistral_api_key=api_key
        )
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Connection Error: {str(e)}"})
        yield history, history, ""
        return

    # Retrieval Step
    context_text = "No context available (General Chat Mode)."
    citations = []
    
    if vectorstore:
        try:
            docs_and_scores = vectorstore.similarity_search_with_score(message, k=4)
            relevant_docs = [doc for doc, score in docs_and_scores if score < 1.0]
            
            if relevant_docs:
                context_text = "\n\n".join([d.page_content for d in relevant_docs])
                citations = sorted(list(set([f"Page {d.metadata.get('page', '?')+1}" for d in relevant_docs])))
        except Exception as e:
            context_text = f"Error retrieving context: {str(e)}"

    # Prompt Engineering
    system_prompt = f"""You are a helpful AI assistant. 
    Answer the user's question using the context provided below. 
    If the answer is not in the context, say so, but you can use your general knowledge if allowed.
    
    Context:
    {context_text}
    """
    
    messages = [
        ("system", system_prompt),
        ("human", message)
    ]

    try:
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = round(time.time() - start_time, 2)
        
        # Format Source Footnote
        source_str = f"\n\n*Sources: {', '.join(citations)}*" if citations else ""
        meta_str = f"\n*(Time: {elapsed}s | Conf: High)*"
        
        final_answer = response.content + source_str + meta_str
        
        # --- STEP 2: Show AI Response ---
        history.append({"role": "assistant", "content": final_answer})
        yield history, history, ""
        
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ LLM Error: {str(e)}"})
        yield history, history, ""

# --- UI Construction ---

with gr.Blocks() as demo:
    
    # State Variables
    user_name = gr.State("")
    vector_store_state = gr.State(None)
    chat_history_state = gr.State([])
    
    # --- PAGE 1: LOGIN ---
    with gr.Column(visible=True) as login_view:
        gr.Markdown("# 👋 Welcome to SMART RAG Assistant")
        name_input = gr.Textbox(label="What should we call you?", placeholder="Enter your name...", scale=2)
        enter_btn = gr.Button("Enter Into Chat 🤖", variant="primary", elem_classes=["primary-btn"])

    # --- PAGE 2: MAIN DASHBOARD ---
    with gr.Column(visible=False) as main_view:
        
        # Header
        with gr.Row(equal_height=True):
            greeting = gr.Markdown("### Hola..! User")
            toggle_dark = gr.Button("🌓 Toggle Dark/Light", size="sm", elem_classes=["dark-mode-btn"])
        
        with gr.Row():
            # LEFT SIDEBAR: Settings & Upload
            with gr.Column(scale=1, variant="panel"):
                gr.Markdown("### ⚙️ Configuration")
                
                api_input = gr.Textbox(
                    label="🔑 Mistral API Key", 
                    type="password", 
                    placeholder="Enter key here...",
                    value="" # Leave empty for security
                )
                
                model_input = gr.Dropdown(
                    choices=["mistral-large-latest", "mistral-small-latest", "open-mixtral-8x7b"], 
                    value="mistral-large-latest", 
                    label="🧠 Model Name"
                )
                
                temp_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.1, 
                    label="🌡️ Creativity (Temperature)"
                )
                
                gr.Markdown("### 📚 Knowledge Base")
                file_uploader = gr.File(
                    label="Upload PDF Documents", 
                    file_count="multiple", 
                    file_types=[".pdf"]
                )
                
                process_btn = gr.Button("⚡ Process Documents", variant="secondary")
                index_status = gr.Markdown("Waiting for files...")

            # RIGHT CONTENT: Chat Interface
            with gr.Column(scale=3):
                
                chatbot = gr.Chatbot(
                    label="Chat Conversation", 
                    height=550,  
                    avatar_images=(None, "https://docs.mistral.ai/img/logo.svg")
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                         
                        placeholder="Ask anything about your documents...",
                        scale=4,
                        
                    )
                    send_btn = gr.Button("📤 Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                    save_btn = gr.Button("💾 Save Conversation", size="sm")
                    download_output = gr.File(label="Download History", visible=False)

    # --- EVENT LISTENERS ---

    # 1. Login Transition
    def login(name):
        return {
            login_view: gr.update(visible=False),
            main_view: gr.update(visible=True),
            greeting: gr.update(value=f"## Hola..! **{name}**")
        }
    
    enter_btn.click(login, inputs=[name_input], outputs=[login_view, main_view, greeting])
    
    # 2. Dark Mode Toggle (JS)
    toggle_dark.click(None, None, None, js=js_func)

    # 3. Document Processing
    process_btn.click(
        fn=process_documents,
        inputs=[file_uploader, api_input],
        outputs=[vector_store_state, index_status]
    )

    # 4. Chat Functionality
    # Trigger on Enter key or Button Click
    chat_triggers = [msg_input.submit, send_btn.click]
    
    for trigger in chat_triggers:
        trigger(
            fn=chat_engine,
            inputs=[msg_input, chat_history_state, vector_store_state, api_input, model_input, temp_slider],
            outputs=[chatbot,chat_history_state, msg_input]
        )

    # 5. Clear Chat
    def clear_chat():
        return [], []
    clear_btn.click(clear_chat, outputs=[chatbot,chat_history_state])

    # 6. Save History
    def save_chat(history):
        path = format_history_for_download(history)
        return gr.update(value=path, visible=True)
    
    save_btn.click(save_chat, inputs=[chat_history_state], outputs=[download_output])

# Launch the app
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css)
