import asyncio
import aiohttp
import argparse
import pydle
import logging
import os
import re
import time
import datetime
import asyncio
import json
from typing import Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="AI is Really Cool")
    parser.add_argument("--server", type=str, default="irc.libera.chat", help="IRC server address")
    parser.add_argument("--port", type=int, default=6667, help="IRC server port")
    parser.add_argument("--nickname", type=str, default="airc", help="Bot nickname")
    parser.add_argument("--channel", type=str, default="#airc-test", help="Channel to join")
    parser.add_argument("--ai-server", type=str, default="http://localhost:5000", help="OpenAI API endpoint")
    parser.add_argument('--level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        default='INFO', help='Set the logging level.')
    return parser.parse_args()

class AsyncOpenAIClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        logging.info(f"Using {base_url} as server")
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def chat_completion(self, 
                            messages: list,
                            model: str = "my-model",
                            temperature: float = 0.7,
                            max_tokens: int = 100) -> Dict[str, Any]:
        """
        Make an async chat completion request
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Request failed with status {response.status}: {error_text}")
                
                return await response.json()
        
        except Exception as e:
            print(f"Error during request: {str(e)}")
            return {"error": str(e)}

def setup_logging(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Logging is set to {level} level.")

class AIRC(pydle.Client):
    def __init__(self, nick, channel, client, burst_limit = 5, rate_limit = 1.0, burst_reset_timeout = 10.0):
        super().__init__(nick)
        self.nick = nick
        self.channel = channel
        self.burst_limit = burst_limit
        self.sent_burst = 0
        self.rate_limit = rate_limit
        self.burst_reset_timeout = burst_reset_timeout
        self.sent_burst = 0  # Track messages sent in burst
        self.last_message_time = None  # Track last message time
        self.system_input = "You are a critical assistant. Give concise and accurate answers in less than 120 characters."
        self._message_queue = asyncio.Queue()
        self._task = asyncio.create_task(self._send_from_queue())
        self.client = client

    async def _send_from_queue(self):
        """Background task that sends queued messages with burst + rate limiting."""
        while True:
            target, message = await self._message_queue.get()

            # If burst is still available, send immediately
            if self.sent_burst < self.burst_limit:
                self.sent_burst += 1
            else:
                await asyncio.sleep(self.rate_limit)  # Apply rate limit
            
            await super().message(target, message)  # Send message
            self.last_message_time = asyncio.get_event_loop().time()  # Update last message timestamp
            
            # Start burst reset countdown after each message
            asyncio.create_task(self._reset_burst_after_inactivity())

    async def _reset_burst_after_inactivity(self):
        """Resets burst counter only if no new messages are sent within timeout."""
        last_time = self.last_message_time
        await asyncio.sleep(self.burst_reset_timeout)  # Wait for inactivity period

        # Only reset if no new messages were sent during the wait
        if self.last_message_time == last_time:
            self.sent_burst = 0
            logging.info("Burst limit reset due to inactivity.")

    async def message(self, target, message):
        """Splits a multi-line message and sends each line separately."""
        for line in message.splitlines():  # Splits on both '\n' and '\r\n'
            if line.strip():  # Ignore empty lines
                await self._message_queue.put((target, line))

    async def on_connect(self):
        logging.debug('on_connect')
        await self.join(self.channel)

    def remove_substring(self, string, substring):
        return string.replace(substring, "")    

    def extract_nick_message(self, input_string):
        # Pattern with capturing groups for nick and message
        pattern = r"^\s*([^\s:]+?)\s*:\s*(.+?)$"
        
        match = re.match(pattern, input_string)
        if match:
            nick = match.group(1)    # First capturing group
            message = match.group(2)  # Second capturing group
            return nick, message
        return None, None  # Return None for both if no match
    
    async def on_message(self, target, source, message):
        if source == self.nick:
            return
        nick, body = self.extract_nick_message(message)
        if nick == self.nick:
            content = None
            if body == "stats":
                content = f"{self.queries} queries handled in {self.processing}s"
            else:
                # Sample messages
                messages = [
                    {"role": "system", "content": self.system_input},
                    {"role": "user", "content": body}
                ]

                # Make the request
                response = await self.client.chat_completion(messages)

                # Extract and print just the assistant's message if available
                if "choices" in response and len(response["choices"]) > 0:
                    content = response["choices"][0]["message"]["content"]
                    print(f"\nAssistant: {content}")

            if content:
                logging.info(f'Sending: {content}')
                await self.message(target, f"{content}")

def remove_substring(string, substring):
    return string.replace(substring, "")
        
async def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging based on the provided level
    setup_logging(args.level)

    async with AsyncOpenAIClient(base_url=args.ai_server) as client:
        bot = AIRC(args.nickname, args.channel, client)
        await bot.connect(args.server, args.port, tls=False)
        await bot.handle_forever()

if __name__ == "__main__":
    asyncio.run(main())
