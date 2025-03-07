from flask import Flask, request, jsonify
import json
import asyncio
import argparse
import pydle
import torch
import logging
from ipex_llm.transformers import AutoModelForCausalLM
import transformers
import os
import re
import time
import datetime
import asyncio
import aiohttp
import json
from typing import Dict, Any
import feedparser
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="AI is Really Cool Server")
    parser.add_argument("--device", type=int, default=0, help="Device # to use for inference. See --device-list")
    #parser.add_argument("--device-list", help="List available devices")
    parser.add_argument('--level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Set the logging level.')
    return parser.parse_args()

def setup_logging(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Logging is set to {level} level.")

class Feed():
    def __init__(self, name, url, poll_limit_min = 30, max_articles=5):
        self.name = name
        self.url = url
        self.poll_limit_min = datetime.timedelta(minutes=poll_limit_min)
        self.last_poll = None
        self.articles = []
        self.max_articles = max_articles
        self.update()

    def update(self):
        now = datetime.datetime.now()
        if self.last_poll is None or (now - self.last_poll) >= self.poll_limit_min:
            logging.info(f"Updating {self.name}")
            feed = feedparser.parse(self.url)
            self.articles = []
            self.last_poll = now

            content = ""
            if len(feed.entries) > 0:
                content += f"Source: {self.name}\n"
            for entry in feed.entries[:self.max_articles]:
                title = entry.get("title")
                if title:
                    content += f"Title: {title}\n"
                link = entry.get("link")
                if link:
                    content += f"Link: {link}\n"
                summary = entry.get("summary")
                if summary:
                    content += f"Summary: {summary}\n"
                published = entry.get("published")
                if published:
                    content += f"Published: {published}\n"
                content += "\n"

                self.articles.append(content)
        else:
            logging.info(f"Not updating {self.name} -- {self.poll_limit_min - (now - self.last_poll)}s remain to refresh.")
        return self.articles


# News RSS Feeds
rss_feeds = [
    Feed(name="BBC World", url="http://feeds.bbci.co.uk/news/world/rss.xml"),
    Feed(name="Reuters World", url="http://feeds.reuters.com/Reuters/worldNews"),
    Feed(name="Al Jazeera", url="https://www.aljazeera.com/xml/rss/all.xml"),
    Feed(name="CNN World", url="http://rss.cnn.com/rss/edition_world.rss"),
    Feed(name="Time", url="https://time.com/feed/"),
    Feed(name="Euronews", url="https://www.euronews.com/rss"),
    Feed(name="FeedX", url="https://feedx.net/rss/ap.xml")
]

# Load an embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Collect news from all sources
documents = []
for feed in rss_feeds:
    documents.extend(feed.articles)

# Step 2: Encode and store news articles into FAISS
doc_vectors = np.array(embedding_model.encode(documents), dtype=np.float32)
index = faiss.IndexFlatL2(doc_vectors.shape[1])  # Initialize FAISS index
index.add(doc_vectors)  # Store news vectors

logging.info(f"Stored {len(doc_vectors)} documents in FAISS index.")

# Step 3: Retrieval function for user queries
def retrieve_documents(query, top_k=2):
    """Retrieve top-k most relevant news articles."""
    query_vector = np.array(embedding_model.encode([query]), dtype=np.float32)
    D, I = index.search(query_vector, top_k)
    retrieved_docs = [documents[i] for i in I[0]]
    return retrieved_docs

# Step 4: Format the RAG prompt
def format_prompt(query, retrieved_docs):
    """Format retrieved documents into a structured RAG prompt."""
    context_str = "\n".join(retrieved_docs)
    prompt = f"""You are an AI assistant with access to world news. Use the Retrieved Context to answer the user's question accurately if relevant, stating which Source provided the information.

## Retrieved Context:
{context_str}

## User Query:
{query}

## Response:
"""
    return prompt


class Chat():
    def __init__(self, device_name):
        super().__init__()
        self.device_name = device_name
        self.system_input = "You are a critical assistant. Give concise and accurate answers in less than 120 characters."
        self.context = None
        self.model_path = 'Intel/neural-chat-7b-v3-3'
        try:
            logging.info(f"Loading tokenizer from: {self.model_path}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad_token to eos_token if needed

            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                load_in_4bit=True,
                                                optimize_model=True,
                                                trust_remote_code=True,
                                                use_cache=True)
            self.model = self.model.half().to(device_name)
        except Exception as e:
            logging.error(f"Loading error: {e}")
            raise Exception(e)

    def remove_substring(self, string, substring):
        return string.replace(substring, "")

    def generate_response(self, text):
        prompt = text
        start = time.time()

        with torch.autocast(self.device_name, dtype=torch.float16):
            inputs = self.tokenizer.encode_plus(
                prompt, 
                add_special_tokens=False,
                return_tensors="pt", 
                max_length=8000,            # Prevent 'Asking to truncate to max_length...'
                padding=True,               # Handles padding automatically
                truncation=True
            )
            input_ids = inputs["input_ids"].to(self.device_name)
            attention_mask = inputs["attention_mask"].to(self.device_name)

            outputs = self.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_length=8000,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            final_outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_outputs = self.remove_substring(final_outputs, prompt).strip()
        
        end = time.time()

        return final_outputs, datetime.timedelta(seconds=end - start)

app = Flask(__name__)

# Basic endpoint for chat completions
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    logging.info('/v1/chat/completions')
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract relevant fields from the request
        model = data.get('model', 'default-model')
        messages = data.get('messages', [])
        temperature = data.get('temperature', 1.0)
        max_tokens = data.get('max_tokens', 2048)
        
        chat = app.config['chat']
        query = messages[-1]['content']

        if re.match(r"^\s*(update|refresh) news\s*$", query, re.IGNORECASE):
            logging.info("New refresh requested")
            # Collect news from all sources
            documents = []
            for feed in rss_feeds:
                documents.extend(feed.update())
            # Step 2: Encode and store news articles into FAISS
            doc_vectors = np.array(embedding_model.encode(documents), dtype=np.float32)
            index = faiss.IndexFlatL2(doc_vectors.shape[1])  # Initialize FAISS index
            index.add(doc_vectors)  # Store news vectors
            logging.info(f"Stored {len(doc_vectors)} documents in FAISS index.")
            response_content = "News refresh requested."
        else:
            logging.info(f"Query: {query}")
            retrieved_docs = retrieve_documents(query)
            rag_prompt = format_prompt(query, retrieved_docs)
            logging.debug(f"RAG prompt: {rag_prompt}")

            # Get AI-generated response
            response_content, _ = chat.generate_response(rag_prompt)

        logging.info(f"Response: {response_content}")
        # Format response in OpenAI-compatible structure
        response = {
            "id": "chatcmpl-" + str(id(data)),  # Simple unique ID
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chat.model_path,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }],
            # "usage": {
            #     "prompt_tokens": len(str(messages).split()),
            #     "completion_tokens": len(response_content.split()),
            #     "total_tokens": len(str(messages).split()) + len(response_content.split())
            # }
        }
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(e)
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error"
            }
        }), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    import time  # Imported here for the timestamp
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging based on the provided level
    setup_logging(args.level)

    if not torch.xpu.is_available():
        logging.error("No XPU available.")
        exit(1)
    device_count = torch.xpu.device_count();
    for i in range(device_count):
        logging.info(f"Device {i}: {torch.xpu.get_device_name(i)} Total memory: {torch.xpu.get_device_properties(i).total_memory}")
    device_name = 'xpu'
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Set environment variables that might help with XPU stability
    os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"

    app.config['chat'] = Chat(device_name)

    app.run(host='0.0.0.0', port=5000, debug=True)