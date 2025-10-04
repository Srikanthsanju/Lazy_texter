import os
import json
import requests
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import chromadb
from chromadb.config import Settings

# --- CONFIGURATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 
GEMINI_MODEL_NAME = "gemini-2.5-flash"
API_BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# --- FLASK SETUP ---
app = Flask(__name__)
CORS(app)

# --- CHROMADB SETUP (Vector Database for RAG) ---
chroma_client = chromadb.Client(Settings(
    anonymized_telemetry=False
))

# Create collections for each chat
timo_collection = chroma_client.get_or_create_collection(
    name="timo_chat",
    metadata={"hnsw:space": "cosine"}
)

shark_collection = chroma_client.get_or_create_collection(
    name="shark_chat",
    metadata={"hnsw:space": "cosine"}
)

CHAT_COLLECTIONS = {
    "Timo": timo_collection,
    "Shark": shark_collection
}

conversation_counter = {"Timo": 0, "Shark": 0}

# --- PERSONAS ---
PERSONAS = {
    "The Strategist": "You are a highly composed, nerd-like professional, inspired by Gary Johnson from Hitman. You speak with quiet confidence, carry an intriguing aura, and balance intellect with subtle humor. You show great chemistry in conversation, sounding observant, analytical, and sharp—always a step ahead without losing charm.",
    "The Visionary": "You are Eddie Morra from Limitless when on NZT: supremely confident, witty, flirtatious, and razor-sharp. Your speech is smooth, fast-paced, and assertive, always laced with clever observations. You sound interesting and magnetic, as if you can read the world in detail and bend it to your will.",
    "The Rebel": "You embody Tyler Durden from Fight Club: a dark philosopher and anarchist. You challenge consumerism and conformity, speaking with passion, madness, and a dangerous kind of charisma. Your tone is raw, rebellious, and existential, with a readiness to fight and a dedication to tearing down illusions.",
    "The Orator": "You speak like an 18th-century Englishman—rhetorical, poetic, and refined. Your language is ornate and formal, filled with metaphors and elevated diction. You sound weary yet profound, as though you are both observing and lamenting the world with philosophical eloquence.",
    "The Conversationalist": "You are a modern Gen Z American, casual yet mature. Your tone is natural, conversational, and slightly witty, but not slang-heavy like rap. You mix straightforward realism with light humor, sounding relatable, grounded, and socially aware without trying too hard."
}

# Persona descriptions for UI
PERSONA_DESCRIPTIONS = {
    "The Strategist": "Inspired by Gary Johnson from Hitman : A step ahead, nerdy intellect, subtle humor.",
    "The Visionary": "Inspired by Eddie Morra from Limitless : Charismatic, magnetic, confident, assertive.",
    "The Rebel": "Inspired by Tyler Durden from Fight Club : Wild, dark idealist, anti-consumerism, fearless.",
    "The Orator": "Inspired by 18th-century Englishmen : Rhetorical, poetic, eloquent, philosophical.",
    "The Conversationalist": "Inspired by modern Gen Z : Casual, witty, relatable, socially aware."
}

# --- RAG FUNCTIONS ---

def store_conversation_rag(chat_id, user_message, persona, stance, response):
    """Store conversation in vector database for semantic search (RAG)."""
    collection = CHAT_COLLECTIONS[chat_id]
    conversation_counter[chat_id] += 1
    
    conv_id = f"{chat_id}_{conversation_counter[chat_id]}"
    
    collection.add(
        documents=[user_message],
        metadatas=[{
            "response": response,
            "persona": persona,
            "stance": stance,
            "timestamp": datetime.now().isoformat()
        }],
        ids=[conv_id]
    )
    
    print(f"[RAG] Stored conversation in {chat_id} - ID: {conv_id}")
    
    current_count = collection.count()
    if current_count > 20:
        all_data = collection.get()
        oldest_ids = all_data['ids'][:current_count - 20]
        collection.delete(ids=oldest_ids)
        print(f"[RAG] Cleaned up {len(oldest_ids)} old conversations from {chat_id}")

def retrieve_relevant_context_rag(chat_id, current_message, top_k=3):
    """RAG: Retrieve semantically similar past conversations using vector search."""
    collection = CHAT_COLLECTIONS[chat_id]
    
    if collection.count() == 0:
        return ""
    
    try:
        results = collection.query(
            query_texts=[current_message],
            n_results=min(top_k, collection.count())
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
        
        context = "\n--- RELEVANT PAST CONVERSATIONS (Retrieved via Semantic Search) ---\n"
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            context += f"\n{i}. Past question: \"{doc}\"\n"
            context += f"   You ({metadata['persona']}) replied: \"{metadata['response']}\"\n"
        
        context += "\n--- END OF RETRIEVED CONTEXT ---\n"
        context += "Note: These conversations were retrieved because they are semantically similar to the current message. Reference them naturally if relevant.\n\n"
        
        print(f"[RAG] Retrieved {len(results['documents'][0])} relevant conversations for {chat_id}")
        return context
        
    except Exception as e:
        print(f"[RAG] Error retrieving context: {str(e)}")
        return ""

def generate_reply(persona_name, user_message, stance, chat_id, response_hint=""):
    """Generates a single reply using the selected persona with RAG-based memory context."""
    
    system_instruction = PERSONAS[persona_name]
    
    memory_context = retrieve_relevant_context_rag(chat_id, user_message)
    
    # Handle response hint (user's draft answer)
    if response_hint:
        # User provided their own answer - rephrase it in persona style
        task_instruction = f"The user wants to reply with: '{response_hint}'. Rephrase this response in your persona's voice and style while keeping the same core meaning. Make it sound natural and characteristic of your persona."
        user_message_with_context = f"Incoming message: {user_message}\n\nUser's draft response: {response_hint}\n\n{task_instruction}"
        memory_instruction = ""
    else:
        # No hint - generate response from scratch
        if memory_context:
            user_message_with_context = f"{memory_context}Current message: {user_message}"
            memory_instruction = "Context from past conversations is shown above. Only reference it if directly relevant to answering the current question. If the current message is a new topic, respond independently."
        else:
            user_message_with_context = user_message
            memory_instruction = ""
    
    input_words = len(user_message.split())
    
    if input_words < 15:
        length_instruction = "Keep your response to 1-2 sentences maximum (under 30 words)."
    elif input_words < 30:
        length_instruction = "Keep your response to 2-3 sentences (under 50 words)."
    else:
        length_instruction = "Keep your response concise (under 80 words)."
    
    if response_hint:
        # When rephrasing, stance is less relevant
        stance_instruction = f"{length_instruction} Write as if texting - use natural conversational language without asterisks, underscores, quotation marks for emphasis, or any special formatting. Output should be plain text ready to copy and paste directly into a chat."
    else:
        stance_instruction = f"Your response MUST {stance.upper()} with the user's message. {length_instruction} Write as if texting - use natural conversational language without asterisks, underscores, quotation marks for emphasis, or any special formatting. Output should be plain text ready to copy and paste directly into a chat. {memory_instruction}"
    
    system_prompt = f"{system_instruction} {stance_instruction}"
    
    payload = {
        "contents": [{"parts": [{"text": user_message_with_context}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }
    
    try:
        response = requests.post(
            API_BASE_URL,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"API Response for {persona_name} (Chat: {chat_id}):")
        print(json.dumps(data, indent=2))
        
        if 'candidates' in data and len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            finish_reason = candidate.get('finishReason', 'UNKNOWN')
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            
            generated_text = ''
            if parts and len(parts) > 0:
                generated_text = parts[0].get('text', '')
            
            if generated_text:
                import re
                cleaned_text = generated_text.strip()
                cleaned_text = cleaned_text.replace('*', '').replace('_', '')
                cleaned_text = re.sub(r'\s"([^"]{1,30})"\s', r' \1 ', cleaned_text)
                
                return {"success": True, "reply": cleaned_text, "persona": persona_name}
            
            if finish_reason == 'MAX_TOKENS':
                error_msg = "Response cut off (MAX_TOKENS)."
            elif finish_reason == 'SAFETY':
                error_msg = "Response blocked by safety filters."
            else:
                error_msg = f"No text generated. Reason: {finish_reason}"
            
            return {"success": False, "error": error_msg}
        else:
            return {"success": False, "error": "No candidates in response."}
            
    except Exception as e:
        print(f"Exception for {persona_name}: {str(e)}")
        return {"success": False, "error": f"Generation Error: {str(e)}"}

# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def serve_html():
    """Serves the index.html file."""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: index.html not found.", 500

@app.route('/generate', methods=['POST'])
def generate_single_reply():
    """Handles single persona reply generation with RAG-based memory."""
    
    if not GEMINI_API_KEY:
        return jsonify({"success": False, "error": "API key missing"}), 500
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        persona_name = data.get('persona', '')
        stance = data.get('stance', 'Agree')
        chat_id = data.get('chat_id', 'Timo')
        response_hint = data.get('response_hint', '').strip()  # Optional user draft
        
        if not user_message:
            return jsonify({"success": False, "error": "Message is required"}), 400
        
        if persona_name not in PERSONAS:
            return jsonify({"success": False, "error": "Invalid persona"}), 400
        
        if chat_id not in CHAT_COLLECTIONS:
            return jsonify({"success": False, "error": "Invalid chat ID"}), 400
        
        result = generate_reply(persona_name, user_message, stance, chat_id, response_hint)
        
        if result.get('success'):
            store_conversation_rag(chat_id, user_message, persona_name, stance, result['reply'])
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in /generate endpoint: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/memory/<chat_id>', methods=['GET'])
def get_memory(chat_id):
    """Retrieve memory for a specific chat from vector database."""
    if chat_id not in CHAT_COLLECTIONS:
        return jsonify({"success": False, "error": "Invalid chat ID"}), 400
    
    collection = CHAT_COLLECTIONS[chat_id]
    
    try:
        all_data = collection.get()
        
        conversations = []
        for doc, metadata, conv_id in zip(all_data['documents'], all_data['metadatas'], all_data['ids']):
            conversations.append({
                "id": conv_id,
                "user_message": doc,
                "response": metadata.get('response', ''),
                "persona": metadata.get('persona', ''),
                "timestamp": metadata.get('timestamp', '')
            })
        
        return jsonify({
            "success": True,
            "chat_id": chat_id,
            "conversations": conversations,
            "count": len(conversations)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/memory/<chat_id>', methods=['DELETE'])
def clear_memory(chat_id):
    """Clear memory for a specific chat from vector database."""
    if chat_id not in CHAT_COLLECTIONS:
        return jsonify({"success": False, "error": "Invalid chat ID"}), 400
    
    collection = CHAT_COLLECTIONS[chat_id]
    
    try:
        all_data = collection.get()
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
        
        conversation_counter[chat_id] = 0
        
        return jsonify({"success": True, "message": f"{chat_id} memory cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/personas', methods=['GET'])
def get_personas():
    """Get list of available personas with descriptions."""
    personas = [
        {"name": name, "description": PERSONA_DESCRIPTIONS[name]}
        for name in PERSONAS.keys()
    ]
    return jsonify({"success": True, "personas": personas})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)