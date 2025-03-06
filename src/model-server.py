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

    def remove_substring(self, string, substring):
        return string.replace(substring, "")

    def generate_response(self, text):
        prompt = f"### System:\n{self.system_input}\n### User:\n{text}\n### Assistant:\n"
        start = time.time()

        with torch.autocast(self.device_name, dtype=torch.float16):
            inputs = self.tokenizer.encode_plus(
                prompt, 
                add_special_tokens=False,
                return_tensors="pt", 
                max_length=1000,            # Prevent 'Asking to truncate to max_length...'
                padding=True,               # Handles padding automatically
                truncation=True
            )
            input_ids = inputs["input_ids"].to(self.device_name)
            attention_mask = inputs["attention_mask"].to(self.device_name)

            outputs = self.model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_length=1000,
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
        logging.info(f"Query: {messages}")
        response_content, _ = chat.generate_response(messages[-1]['content'])
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