# gemini_mistral_api.py

# ==============================================================================
# Gemini-Mistral Unified AI Gateway Server
# ==============================================================================
# This server acts as a single endpoint for both Mistral and Google Gemini APIs.
# It intelligently routes requests based on the 'model' parameter.
# - Mistral requests are handled asynchronously for high throughput.
# - Gemini requests leverage a robust, thread-safe key manager with auto-retry
#   and key-revival logic, bridged to the async world via asyncio.to_thread.
# - Features a real-time monitoring dashboard for both services.
# - Handles multimodal (image) inputs in OpenAI format for both backends.
# - Includes a dedicated OCR endpoint for Mistral's OCR models.
# - Adds Mistral Batch Inference and File Management endpoints.
# ==============================================================================

import asyncio
import os
import json
import time
import base64
import re
import io
import threading
import queue
import contextlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict

# --- Third-party libraries ---
import pandas as pd
import requests
from aiohttp import web, WSMsgType
from mistralai import Mistral
from mistralai import File # Corrected import for batch operations
from PIL import Image
from rich.console import Console

# --- Gemini-Specific Imports (requires 'pip install google-generativeai') ---
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# --- Rich Console for Better Logging ---
console = Console()


# ==============================================================================
# 1. Custom Gemini-Specific Exceptions
# ==============================================================================
class NoAvailableKeysError(Exception):
    """Raised when no Gemini API keys are available and none are expected to become available."""
    pass

class ApiKeyQuotaExceededError(Exception):
    """Custom exception raised when a Gemini API key hits its rate limit or quota."""
    pass


# ==============================================================================
# 2. Gemini: The Thread-Safe Key Manager (REFACTORED)
# ==============================================================================
class ApiKeyManager:
    """
    A thread-safe manager for handling a pool of Google Gemini API keys.
    This class is designed to run in its own threads and is safe to use from an
    asyncio event loop via `asyncio.to_thread`.

    *** REFACTORED LOGIC ***
    - Keys are no longer locked exclusively for a single request.
    - Active keys are stored in a list and selected via a round-robin strategy,
      allowing for concurrent use of the same API key.
    - A key is only "retired" reactively when an ApiKeyQuotaExceededError is reported.
    - If all keys are retired, new requests will block until a key is revived.
    """
    def __init__(
        self,
        sheet_url: str,
        revival_delay_seconds: int = 90,
        max_failures: int = 4,
        reload_interval: int = 1800,  # 30 minutes
    ):
        self.sheet_url = sheet_url
        self.revival_delay_seconds = revival_delay_seconds
        self.max_failures = max_failures
        self.reload_interval = reload_interval

        # --- Refactored State Management ---
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock) # For waiting when no keys are active
        self._active_keys: List[Dict] = []
        self._retired_keys: List[tuple[float, Dict]] = []
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._current_key_index = 0
        self._all_known_tokens = set()

        self._stop_event = threading.Event()

        self._load_and_initialize_keys()

        self._revival_thread = threading.Thread(target=self._revive_keys_periodically, daemon=True)
        self._revival_thread.start()

        self._reload_thread = threading.Thread(target=self._reload_keys_periodically, daemon=True)
        self._reload_thread.start()

        console.print(f"[Gemini KeyManager] Initialized with [bold green]{len(self._active_keys)}[/bold green] active keys, ready for concurrent use.")

    def _load_api_keys_from_sheet(self) -> list:
        console.print("[Gemini KeyManager] Fetching API keys from Google Sheets...")
        try:
            response = requests.get(self.sheet_url)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip()
            if 'Token' not in df.columns and 'Name' not in df.columns:
                raise ValueError("Gemini CSV must contain 'Token' and 'Name' columns.")
            df = df.dropna(subset=['Token', 'Name']).drop_duplicates()
            return df[['Token', 'Name']].rename(columns={'Token': 'token', 'Name': 'name'}).to_dict('records')
        except Exception as e:
            raise RuntimeError(f"Failed to load Gemini API keys: {e}")

    def _load_and_initialize_keys(self):
        """Loads keys from the sheet and populates the initial active key list."""
        try:
            new_keys_info = self._load_api_keys_from_sheet()
            with self._lock:
                # Add only new keys that we haven't seen before
                newly_added = [info for info in new_keys_info if info['token'] not in self._all_known_tokens]
                for info in newly_added:
                    self._active_keys.append(info)
                    self._all_known_tokens.add(info['token'])
                
                if newly_added:
                    console.print(f"[Gemini KeyManager] Loaded [green]{len(newly_added)}[/green] new keys.")
                    # Notify any waiting threads that new keys are available
                    self._condition.notify_all()
            
            if not self._all_known_tokens:
                 raise ValueError("No Gemini API keys loaded. Please check the sheet URL and content.")

        except Exception as e:
             console.print(f"[Gemini KeyManager] [red]Key loading/reloading failed:[/red] {e}")

    def _revive_keys_periodically(self):
        """Periodically checks the retired keys and moves them back to active status if their cooldown has passed."""
        while not self._stop_event.wait(5):
            with self._lock:
                now = time.time()
                keys_to_revive = []
                remaining_retired = []

                for retirement_time, key_info in self._retired_keys:
                    key_name = key_info["name"]
                    if now - retirement_time >= self.revival_delay_seconds:
                        if self._failure_counts.get(key_name, 0) < self.max_failures:
                            keys_to_revive.append(key_info)
                        # else: key is permanently failed, do not add to remaining_retired
                    else:
                        remaining_retired.append((retirement_time, key_info))
                
                if keys_to_revive:
                    self._retired_keys = remaining_retired
                    for key_info in keys_to_revive:
                        self._active_keys.append(key_info)
                        console.print(f"[Gemini KeyManager] ✅ Revived key [cyan]{key_info['name']}[/cyan]. Active keys: {len(self._active_keys)}")
                    # A key was revived, notify any waiting threads
                    self._condition.notify_all()


    def _reload_keys_periodically(self):
        while not self._stop_event.wait(self.reload_interval):
            self._load_and_initialize_keys()

    def select_key(self) -> Dict[str, Any]:
        """
        Selects an active key using a round-robin strategy.
        If no keys are available, this method blocks until one is revived or loaded.
        """
        with self._lock:
            # Wait until there's at least one active key
            while not self._active_keys:
                console.print("[Gemini KeyManager] No active keys available. Request is waiting for a key to be revived...", style="yellow")
                self._condition.wait() # This releases the lock and waits for a notify() call

            # Round-robin selection
            self._current_key_index = (self._current_key_index + 1) % len(self._active_keys)
            key_info = self._active_keys[self._current_key_index]
            return key_info

    def retire_key(self, key_info: Dict[str, Any]):
        """
        Reactively retires a key that has failed, moving it from the active pool to the retired list.
        """
        with self._lock:
            key_token = key_info['token']
            key_name = key_info['name']
            
            # Ensure the key is actually in the active list before trying to remove
            original_len = len(self._active_keys)
            self._active_keys = [k for k in self._active_keys if k['token'] != key_token]
            if len(self._active_keys) == original_len:
                # Key was already retired by another concurrent request, do nothing.
                return

            self._failure_counts[key_name] += 1
            fail_count = self._failure_counts[key_name]
            
            if fail_count >= self.max_failures:
                console.print(f"[Gemini KeyManager] 🔥 Permanently removed key [red]{key_name}[/red] after {fail_count} failures.")
                # The key is not added to the retired list, effectively removing it forever.
            else:
                self._retired_keys.append((time.time(), key_info))
                active_count = len(self._active_keys)
                cooldown_count = len(self._retired_keys)
                console.print(f"[Gemini KeyManager]  retiring key [red]{key_name}[/red] for {self.revival_delay_seconds}s (failures: {fail_count}). Active: {active_count}, Cooldown: {cooldown_count}")

    def get_active_key_count(self) -> int:
        with self._lock:
            return len(self._active_keys)

    def shutdown(self):
        console.print("[Gemini KeyManager] Shutting down...")
        self._stop_event.set()
        with self._lock:
            self._condition.notify_all() # Wake up any waiting threads so they can exit
        if self._revival_thread.is_alive(): self._revival_thread.join(timeout=1)
        if self._reload_thread.is_alive(): self._reload_thread.join(timeout=1)
        console.print("[Gemini KeyManager] Shutdown complete.")

# ==============================================================================
# 3. Gemini: The Stateless Generator Client (REFACTORED)
# ==============================================================================
class GeneratorGemini:
    """
    A self-sufficient client for the Google Gemini API.
    It now selects a key for concurrent use and reactively retires it upon failure.
    If no keys are available, it will wait indefinitely until one is revived.
    """
    def __init__(self, api_key_manager: 'ApiKeyManager', model_name="gemini-1.5-pro-latest", temperature=0.2, max_new_tokens=8192):
        self.api_key_manager = api_key_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        if not genai:
             raise ImportError("Please install 'google-generativeai' to use the Gemini backend.")
        self.genai = genai

    def _prepare_api_request(self, messages: list, images: Optional[List[Image.Image]] = None):
        if not messages:
            raise ValueError("'messages' must be provided.")

        history, last_message = messages[:-1], messages[-1]

        formatted_history = []
        for m in history:
            role = "model" if m["role"] == "assistant" else "user"
            content_part = m.get("content", "")
            if isinstance(content_part, str):
                formatted_history.append({"role": role, "parts": [content_part]})

        prompt_content = []
        last_content = last_message.get('content', '')
        if isinstance(last_content, str) and last_content:
            prompt_content.append(last_content)
        if images:
            for img in images:
                prompt_content.append(img)
        return prompt_content, formatted_history

    def generate(self, messages: list,
                 images: Optional[List[Image.Image]] = None,
                 model_name: str = None,
                 max_new_tokens: int = None, temperature: float = None) -> str:
        """
        Generates a response from the Gemini API. If all keys are temporarily exhausted,
        this method will block and wait until a key becomes available.
        It uses a selected key concurrently and only retires it on failure.
        """
        # This loop will continue until a request succeeds with a key.
        while True:
            # Select a key from the manager. This call blocks if no keys are active.
            # It does NOT lock the key for exclusive use.
            key_info = self.api_key_manager.select_key()
            
            try:
                console.print(f"[GeneratorGemini] Attempting request with key '[cyan]{key_info['name']}[/cyan]'.", style="yellow")
                self.genai.configure(api_key=key_info['token'])

                generation_config = {
                    "max_output_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                    "temperature": temperature if temperature is not None else self.temperature,
                }
                model_to_use = model_name or self.model_name
                t1 = time.time()
                model = self.genai.GenerativeModel(model_to_use, generation_config=generation_config)
                t2 = time.time()
                print(f"[GeneratorGemini] Model '{model_to_use}' initialized in {t2 - t1:.2f} seconds.")

                prompt_content, api_history = self._prepare_api_request(messages, images)
                if not prompt_content:
                    raise ValueError("Cannot send a message with no content (text or images).")

                chat = model.start_chat(history=api_history)
                response = chat.send_message(prompt_content, stream=False)

                console.print(f"[GeneratorGemini] Request successful with key '[cyan]{key_info['name']}[/cyan]'.", style="green")
                return response.text  # Success! Exit the while loop.

            except Exception as e:
                error_str = str(e).lower()
                # Check for specific quota/rate limit errors
                if any(keyword in error_str for keyword in ["quota", "rate limit", "exceeded", "expired", "permission_denied"]):
                    # This specific key is exhausted. Tell the manager to retire it.
                    console.print(f"[GeneratorGemini] Quota error on key '[cyan]{key_info['name']}[/cyan]'. Retiring it.", style="bold red")
                    self.api_key_manager.retire_key(key_info)
                    # The 'continue' will cause the while loop to try again, selecting a new key.
                    continue
                else:
                    # For other, unexpected API errors, we don't necessarily blame the key.
                    # We re-raise the exception to be handled by the main request handler.
                    console.print(f"[GeneratorGemini] Unexpected API error with key '[cyan]{key_info['name']}[/cyan]': {e}", style="bold red")
                    raise  # Propagate other errors up.
                
# ==============================================================================
# 4. Server Configuration and Global State
# ==============================================================================

# --- Configuration ---
MISTRAL_SHEET_URL = os.getenv("MISTRAL_SHEET_URL", "https://docs.google.com/spreadsheets/d/1NAlj7OiD9apH3U47RLJK0en1wLSW78X5zqmf6NmVUA4/export?format=csv&gid=0")
GEMINI_SHEET_URL = os.getenv("GEMINI_SHEET_URL", "https://docs.google.com/spreadsheets/d/1Sa2OZ6DSWET1zA70z8CiwReCsNR5OscsZKPN7OxVo2I/gviz/tq?tqx=out:csv&gid=0")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 9600))

# Performance and Behavior
MAX_CONCURRENT_REQUESTS = 100
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3 # For Mistral calls
RETRY_DELAYS = [1.0, 2.0, 3.0]

# --- Model Routing Configuration ---
DEFAULT_MISTRAL_MODEL = "mistral-medium-latest"
DEFAULT_MISTRAL_VISION_MODEL = "pixtral-large-latest"
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest"

# Gemini and related Google models
GEMINI_MODEL_NAMES = {
    # Original Models
    "gemini-1.5-pro-latest",
    "gemini-pro",
    "gemini-1.0-pro",

    # Gemini 2.5 Series
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",

    # Gemini 2.0 Series
    "gemini-2.0-flash",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-lite",

    # Deprecated Gemini 1.5 Models
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",

    # Imagen Models
    "imagen-4-standard",
    "imagen-4-ultra",
    "imagen-3",

    # Other Google Models
    "veo-2",
    "gemma-3",
    "gemma-3-27b-it",
    "gemma-3n",
    "gemini-embedding-experimental-03-07",

    # OpenAI compatible names routed to Gemini
    "gpt-4o",
    "gpt-4-vision-preview",
}

# Mistral models with vision/multimodal capabilities
MISTRAL_VISION_MODELS = {
    # Pixtral Models (explicitly vision)
    "pixtral-12b-2409",
    "pixtral-large-latest",
    "pixtral-large-2411",

    # Other Multimodal Models
    "mistral-medium-latest",  # "frontier-class multimodal model"
    "mistral-medium-2505",
    "mistral-small-2503",     # "image understanding capabilities"
    
    # OCR Models (vision-related)
    "mistral-ocr-latest",
    "mistral-ocr-2505",
}
ALL_VISION_MODELS = GEMINI_MODEL_NAMES.union(MISTRAL_VISION_MODELS)

# --- Global State Variables ---
mistral_key_queue: asyncio.Queue = None
mistral_client_pool: Dict[str, Mistral] = {}
mistral_key_last_used = defaultdict(float)
mistral_rate_limit_delay = 1.0
mistral_blacklisted_keys = set()
gemini_key_manager: ApiKeyManager = None
gemini_generator: GeneratorGemini = None
request_semaphore: asyncio.Semaphore = None

# ==============================================================================
# 5. Real-Time Metrics & Dashboard
# ==============================================================================
class RealTimeMetrics:
    def __init__(self):
        self.active_requests = {}
        self.request_history = deque(maxlen=1000)
        self.stats = {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "active_count": 0, "avg_response_time": 0.0, "requests_per_minute": 0.0,
            "peak_concurrent": 0, "uptime": time.time(),
            "mistral_keys_available": 0, "gemini_keys_available": 0,
            "current_load": 0.0, "multimodal_requests": 0, "vision_requests": 0
        }
        self.response_times = deque(maxlen=100)
        self.minute_buckets = defaultdict(int)
        self.error_types = defaultdict(int)
        self.model_usage = defaultdict(int)
        self.websocket_clients = set()
    def start_request(self, request_id: str, model: str, has_images: bool = False):
        current_time = time.time()
        self.active_requests[request_id] = {"start_time": current_time, "model": model, "has_images": has_images}
        self.stats["active_count"] = len(self.active_requests)
        self.stats["peak_concurrent"] = max(self.stats["peak_concurrent"], self.stats["active_count"])
        self.stats["current_load"] = (self.stats["active_count"] / MAX_CONCURRENT_REQUESTS) * 100
        self.model_usage[model] += 1
        if has_images: self.stats["multimodal_requests"] += 1
        if model in ALL_VISION_MODELS: self.stats["vision_requests"] += 1
    def end_request(self, request_id: str, success: bool, error_type: str = None):
        if request_id not in self.active_requests: return
        current_time = time.time()
        start_time = self.active_requests[request_id]["start_time"]
        response_time = current_time - start_time
        self.stats["total_requests"] += 1
        if success: self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
            if error_type: self.error_types[error_type] += 1
        self.response_times.append(response_time)
        if self.response_times: self.stats["avg_response_time"] = sum(self.response_times) / len(self.response_times)
        self.request_history.append({
            "timestamp": current_time, "response_time": response_time, "success": success,
            "model": self.active_requests[request_id]["model"], "has_images": self.active_requests[request_id]["has_images"], "error_type": error_type
        })
        minute_key = int(current_time // 60)
        self.minute_buckets[minute_key] += 1
        recent_minutes = [self.minute_buckets.get(minute_key - i, 0) for i in range(5)]
        self.stats["requests_per_minute"] = sum(recent_minutes) / 5.0
        del self.active_requests[request_id]
        self.stats["active_count"] = len(self.active_requests)
        self.stats["current_load"] = (self.stats["active_count"] / MAX_CONCURRENT_REQUESTS) * 100
    def get_dashboard_data(self):
        current_time = time.time()
        uptime_seconds = current_time - self.stats["uptime"]
        recent_requests = list(self.request_history)
        if mistral_key_queue: self.stats["mistral_keys_available"] = mistral_key_queue.qsize()
        if gemini_key_manager: self.stats["gemini_keys_available"] = gemini_key_manager.get_active_key_count()
        return {
            "stats": {
                **self.stats, "uptime_formatted": str(timedelta(seconds=int(uptime_seconds))),
                "success_rate": (self.stats["successful_requests"] / max(self.stats["total_requests"], 1)) * 100,
                "error_rate": (self.stats["failed_requests"] / max(self.stats["total_requests"], 1)) * 100,
                "multimodal_percentage": (self.stats["multimodal_requests"] / max(self.stats["total_requests"], 1)) * 100
            },
            "active_requests": [{"id": req_id, "duration": current_time - req_data["start_time"], "model": req_data["model"], "has_images": req_data.get("has_images", False)} for req_id, req_data in self.active_requests.items()],
            "recent_performance": [{"timestamp": r["timestamp"] * 1000, "response_time": r["response_time"] * 1000, "success": r["success"], "has_images": r.get("has_images", False)} for r in recent_requests[-50:]],
            "error_breakdown": dict(self.error_types), "model_usage": dict(self.model_usage), "timestamp": current_time * 1000
        }

metrics = RealTimeMetrics()

# ==============================================================================
# 6. Helper Functions and Parsers
# ==============================================================================

def load_mistral_keys(url: str) -> List[str]:
    try:
        df = pd.read_csv(url, skiprows=1, usecols=[0], header=None, dtype=str)
        keys = df[0].dropna().str.strip()
        return [k for k in keys if k]
    except Exception as e:
        console.print(f"❌ Error loading Mistral API keys: {e}", style="red")
        return []

def init_mistral_client_pool(api_keys: List[str]) -> None:
    global mistral_client_pool
    for key in api_keys:
        mistral_client_pool[key] = Mistral(api_key=key)
    console.print(f"🔑 [Mistral] Initialized {len(mistral_client_pool)} pre-built clients.")

def is_base64_image(data_url: str) -> bool:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"): return False
    try:
        base64.b64decode(data_url.split("base64,")[1], validate=True)
        return True
    except Exception: return False

def parse_openai_messages(messages: List[Dict]) -> tuple[List[Dict], List[Image.Image], bool]:
    """Parses messages to separate text, create PIL Images, and format for different backends."""
    cleaned_messages, pil_images, has_images = [], [], False
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if is_base64_image(url):
                        has_images = True
                        try:
                            image_data = base64.b64decode(url.split("base64,")[1])
                            pil_images.append(Image.open(io.BytesIO(image_data)).convert('RGB'))
                        except Exception as e:
                            console.print(f"[Parser] Failed to decode Base64 image: {e}", style="red")
            cleaned_messages.append({"role": role, "content": " ".join(text_parts)})
        else:
            cleaned_messages.append(msg)
    return cleaned_messages, pil_images, has_images

def generate_simple_id(length_bytes: int = 9) -> str:
    rand_bytes = os.urandom(length_bytes)
    return base64.urlsafe_b64encode(rand_bytes).rstrip(b'=').decode("ascii")

def create_openai_response(response_text: str, model: str) -> Dict[str, Any]:
    return {
        "id": generate_simple_id(16),  # Just the base64 ID
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

def create_mistral_response_dict(chat_response, model: str) -> Dict[str, Any]:
    response = { "id": getattr(chat_response, "id", ""), "object": "chat.completion", "created": int(time.time()), "model": model, "choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    if hasattr(chat_response, "choices") and chat_response.choices:
        choice = chat_response.choices[0]
        response["choices"] = [{"index": 0, "message": {"role": "assistant", "content": getattr(choice.message, "content", "")}, "finish_reason": getattr(choice, "finish_reason", "stop")}]
    if hasattr(chat_response, "usage"):
        usage = chat_response.usage
        response["usage"] = {"prompt_tokens": getattr(usage, "prompt_tokens", 0), "completion_tokens": getattr(usage, "completion_tokens", 0), "total_tokens": getattr(usage, "total_tokens", 0)}
    return response

# ==============================================================================
# 7. Core API Call Processors
# ==============================================================================

# Helper for Mistral key management
async def _acquire_mistral_client_key():
    """Acquires a Mistral API key from the queue, handling rate limits and blacklisting."""
    while True:
        api_key = await mistral_key_queue.get()
        if api_key in mistral_blacklisted_keys:
            console.print(f"🚫 [Mistral Key] Skipping blacklisted key {api_key[:10]}...", style="red")
            # Don't put blacklisted key back, just discard and get another
            continue

        current_time = time.time()
        last_used = mistral_key_last_used.get(api_key, 0)

        # Check if the key has been used too recently
        if current_time - last_used < mistral_rate_limit_delay:
            # If so, put it back to the end of the queue and try another one
            mistral_key_queue.put_nowait(api_key)
            # Give a very short breath to avoid busy-waiting if all keys are rate-limited
            await asyncio.sleep(0.01)
            continue
        
        # If we reach here, we have a usable key
        mistral_key_last_used[api_key] = current_time
        return api_key

def _release_mistral_client_key(api_key: str):
    """Releases a Mistral API key back to the queue."""
    if api_key and api_key not in mistral_blacklisted_keys:
        mistral_key_queue.put_nowait(api_key)

def _blacklist_mistral_client_key(api_key: str):
    """Blacklists a Mistral API key, removing it from active use."""
    if api_key not in mistral_blacklisted_keys:
        mistral_blacklisted_keys.add(api_key)
        console.print(f"🚫 [Mistral] Blacklisting key: {api_key[:10]}...", style="red")
        # Do NOT put it back in the queue.

async def process_gemini_call(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"req_gemini_{int(time.time() * 1000)}"
    model = request_data.get("model", DEFAULT_GEMINI_MODEL)
    messages = request_data.get("messages", [])

    if not messages: return {"status": "error", "error_message": "'messages' field is required"}

    cleaned_messages, pil_images, has_images = parse_openai_messages(messages)
    metrics.start_request(request_id, model, has_images)
    
    try:
        response_text = await asyncio.to_thread(
            gemini_generator.generate,
            messages=cleaned_messages, images=pil_images, model_name=model,
            max_new_tokens=request_data.get("max_tokens"), temperature=request_data.get("temperature")
        )
        response_dict = create_openai_response(response_text, model)
        metrics.end_request(request_id, True)
        return {"status": "success", "data": response_dict}
    except Exception as e:
        console.print(f"[Gemini Processor] [bold red]Error processing request {request_id}: {e}[/bold red]")
        console.print_exception()
        metrics.end_request(request_id, False, "gemini_api_error")
        return {"status": "error", "error_message": f"Gemini request failed: {e}"}

async def process_mistral_call(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"req_mistral_{int(time.time() * 1000)}"
    requested_model = request_data.get("model", DEFAULT_MISTRAL_MODEL)
    api_key = None
    model = requested_model
    has_images = False
    
    try:
        messages = request_data.get("messages", [])
        if not messages:
            metrics.start_request(request_id, model)
            metrics.end_request(request_id, False, "invalid_request")
            return {"status": "error", "error_message": "'messages' field is required"}

        # Mistral needs its own message format conversion
        converted_messages = []
        for msg in messages:
            if isinstance(msg.get('content'), list):
                # For mistral vision, convert to its own format
                has_images = True
                mistral_content = []
                for part in msg['content']:
                    if part['type'] == 'text':
                        mistral_content.append({"type": "text", "text": part.get("text", "")})
                    elif part['type'] == 'image_url' and is_base64_image(part.get("image_url", {}).get("url")):
                        mistral_content.append({"type": "image_url", "image_url": part['image_url']['url']})
                converted_messages.append({"role": msg['role'], "content": mistral_content})
            else:
                 converted_messages.append(msg)
        
        if has_images and requested_model not in MISTRAL_VISION_MODELS:
            model = DEFAULT_MISTRAL_VISION_MODEL
            console.print(f"📷 [Mistral] Request has images, auto-switching to vision model '{model}'")
        else:
            model = requested_model

        metrics.start_request(request_id, model, has_images)
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            async with request_semaphore:
                api_key = None # Ensure api_key is None for each attempt's acquisition
                try:
                    api_key = await _acquire_mistral_client_key()
                    client = mistral_client_pool[api_key]
                    
                    params = {"model": model, "messages": converted_messages}
                    for p in ["temperature", "max_tokens", "top_p"]:
                        if p in request_data: params[p] = request_data[p]
                    
                    chat_response = await asyncio.to_thread(client.chat.complete, **params)
                    
                    metrics.end_request(request_id, True)
                    response_dict = create_mistral_response_dict(chat_response, requested_model)
                    _release_mistral_client_key(api_key) # Release on success
                    return {"status": "success", "data": response_dict}
                    
                except Exception as e:
                    error_message, last_error = str(e).lower(), str(e)
                    if "unauthorized" in error_message or ("invalid" in error_message and "key" in error_message):
                        if api_key: _blacklist_mistral_client_key(api_key) # Blacklist on auth error
                        # The continue will ensure a new key is acquired in the next loop iteration
                        continue
                    else:
                        if api_key: _release_mistral_client_key(api_key) # Release on other errors
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                            console.print(f"❌ [Mistral] API error (attempt {attempt+1}): {str(e)[:100]}... Retrying in {delay}s.", style="yellow")
                            await asyncio.sleep(delay)
                            continue
                    break # Break retry loop on unrecoverable error

        metrics.end_request(request_id, False, "mistral_api_error")
        return {"status": "error", "error_message": f"Mistral request failed after {MAX_RETRIES} attempts: {last_error}"}
    except Exception as e:
        # If an error happens *before* key acquisition or during key handling
        metrics.end_request(request_id, False, "system_error")
        console.print_exception()
        return {"status": "error", "error_message": f"System error in Mistral processor: {e}"}

async def process_mistral_ocr_call(request_data: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"req_ocr_{int(time.time() * 1000)}"
    model = request_data.get("model", "mistral-ocr-latest")
    image_url = request_data.get("image")
    api_key = None # Initialize to None for outer scope

    metrics.start_request(request_id, model, has_images=True)

    if not image_url or not is_base64_image(image_url):
        metrics.end_request(request_id, False, "invalid_request")
        return {"status": "error", "error_message": "A valid base64 'image' data URL is required."}

    try:
        last_error = None
        for attempt in range(MAX_RETRIES):
            async with request_semaphore:
                api_key = None # Ensure api_key is None for each attempt's acquisition
                try:
                    api_key = await _acquire_mistral_client_key()
                    client = mistral_client_pool[api_key]

                    # The document payload for the OCR API
                    document_payload = {
                        "type": "image_url",
                        "image_url": image_url
                    }
                    
                    # Call the OCR API
                    ocr_response = await asyncio.to_thread(
                        client.ocr.process,
                        model=model,
                        document=document_payload,
                        include_image_base64=request_data.get("include_image_base64", False)
                    )
                    
                    metrics.end_request(request_id, True)
                    _release_mistral_client_key(api_key) # Release on success
                    return {"status": "success", "data": ocr_response.dict()}
                    
                except Exception as e:
                    error_message, last_error = str(e).lower(), str(e)
                    if "unauthorized" in error_message or ("invalid" in error_message and "key" in error_message):
                        if api_key: _blacklist_mistral_client_key(api_key)
                        continue
                    else:
                        if api_key: _release_mistral_client_key(api_key)
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                            console.print(f"❌ [Mistral OCR] API error (attempt {attempt+1}): {str(e)[:100]}... Retrying in {delay}s.", style="yellow")
                            await asyncio.sleep(delay)
                            continue
                    break

        metrics.end_request(request_id, False, "mistral_ocr_api_error")
        return {"status": "error", "error_message": f"Mistral OCR request failed after {MAX_RETRIES} attempts: {last_error}"}
    except Exception as e:
        metrics.end_request(request_id, False, "system_error")
        console.print_exception()
        return {"status": "error", "error_message": f"System error in Mistral OCR processor: {e}"}

async def process_mistral_file_upload(request_id: str, file_name: str, file_content: bytes, purpose: str) -> Dict[str, Any]:
    api_key = None
    try:
        api_key = await _acquire_mistral_client_key()
        client = mistral_client_pool[api_key]

        console.print(f"[{request_id}] ⬆️  Uploading file '{file_name}' for purpose '{purpose}' using key ending in ...{api_key[-4:]}")
        upload_response = await asyncio.to_thread(
            client.files.upload,
            file=File(file_name=file_name, content=file_content),
            purpose=purpose
        )
        file_id = upload_response.id
        console.print(f"[{request_id}] ✅ File uploaded successfully. File ID: {file_id}", style="green")

        if purpose == "batch":
            max_retries = 5
            initial_delay = 2.0
            console.print(f"[{request_id}] ⏳ Polling for file readiness (ID: {file_id})...", style="yellow")
            for attempt in range(max_retries):
                try:
                    await asyncio.to_thread(client.files.retrieve, file_id=file_id)
                    console.print(f"[{request_id}] ✅ File is indexed and ready.", style="green")
                    
                    # Add the API key to the successful response
                    response_data = upload_response.model_dump()
                    response_data['api_key_for_session'] = api_key
                    _release_mistral_client_key(api_key) # Release key back to pool
                    return {"status": "success", "data": response_data}
                except Exception as e:
                    if "404" in str(e) and attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        console.print(f"[{request_id}] File not indexed. Retrying in {delay:.1f}s...", style="yellow")
                        await asyncio.sleep(delay)
                    else:
                        raise # Re-raise final or unexpected error
            raise RuntimeError(f"File readiness check timed out for {file_id}.")

        # For non-batch purposes
        response_data = upload_response.model_dump()
        response_data['api_key_for_session'] = api_key
        _release_mistral_client_key(api_key)
        return {"status": "success", "data": response_data}

    except Exception as e:
        console.print(f"[{request_id}] [bold red]Error processing file upload: {e}[/bold red]")
        console.print_exception()
        if api_key:
            _blacklist_mistral_client_key(api_key)
        return {"status": "error", "error_message": f"Mistral file upload process failed: {e}"}
     
async def process_mistral_file_download(request_id: str, file_id: str):
    api_key = None
    try:
        for attempt in range(MAX_RETRIES):
            api_key = None
            try:
                api_key = await _acquire_mistral_client_key()
                client = mistral_client_pool[api_key]
                
                # The download returns a stream-like object, which we'll read into bytes
                output_file_stream = await asyncio.to_thread(client.files.download, file_id=file_id)
                
                # Mistral SDK's client.files.download returns a httpx.Response, which has .iter_bytes()
                # or .read(). For simplicity, let's read it fully.
                content_bytes = await asyncio.to_thread(output_file_stream.read)
                
                _release_mistral_client_key(api_key)
                return {"status": "success", "data": content_bytes}
            except Exception as e:
                error_message = str(e).lower()
                if "unauthorized" in error_message or ("invalid" in error_message and "key" in error_message):
                    if api_key: _blacklist_mistral_client_key(api_key)
                else:
                    if api_key: _release_mistral_client_key(api_key)
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                    console.print(f"❌ [Mistral File Download] API error (attempt {attempt+1}): {str(e)[:100]}... Retrying in {delay}s.", style="yellow")
                    await asyncio.sleep(delay)
                    continue
                raise
    except Exception as e:
        console.print(f"[Mistral File Download Processor] [bold red]Error downloading file {request_id} for ID {file_id}: {e}[/bold red]")
        console.print_exception()
        return {"status": "error", "error_message": f"Mistral file download failed: {e}"}

async def process_mistral_batch_job_create(request_id: str, batch_data: Dict[str, Any]) -> Dict[str, Any]:
    # Check if a specific key was passed for this session
    session_api_key = batch_data.pop('api_key_for_session', None)
    api_key_acquired = False
    
    try:
        if session_api_key:
            api_key = session_api_key
            console.print(f"[{request_id}] 📌 Using pinned API key for batch session ending in ...{api_key[-4:]}")
        else:
            # Fallback for old behavior, though not recommended for batch
            api_key = await _acquire_mistral_client_key()
            api_key_acquired = True

        client = mistral_client_pool[api_key]
        
        response = await asyncio.to_thread(
            client.batch.jobs.create,
            input_files=batch_data.get("input_files"),
            model=batch_data.get("model"),
            endpoint=batch_data.get("endpoint"),
            metadata=batch_data.get("metadata")
        )
        
        # Only release the key if we acquired it from the pool in this function
        if api_key_acquired:
            _release_mistral_client_key(api_key)
            
        return {"status": "success", "data": response.model_dump()}
            
    except Exception as e:
        console.print(f"[Mistral Batch Create Processor] [bold red]Error processing batch job creation {request_id}: {e}[/bold red]")
        console.print_exception()
        # Handle key release/blacklist on error
        if session_api_key:
            # If a pinned key fails, it's definitely bad. Blacklist it.
            _blacklist_mistral_client_key(session_api_key)
        elif api_key_acquired:
            # If a freshly acquired key fails, handle it based on error type
            error_message = str(e).lower()
            if "unauthorized" in error_message or ("invalid" in error_message and "key" in error_message):
                _blacklist_mistral_client_key(api_key)
            else:
                 _release_mistral_client_key(api_key)
        
        return {"status": "error", "error_message": f"Mistral batch job creation failed: {e}"}
    
async def process_mistral_batch_job_get(request_id: str, job_id: str) -> Dict[str, Any]:
    api_key = None
    try:
        for attempt in range(MAX_RETRIES):
            api_key = None
            try:
                api_key = await _acquire_mistral_client_key()
                client = mistral_client_pool[api_key]
                
                response = await asyncio.to_thread(client.batch.jobs.get, job_id=job_id)
                _release_mistral_client_key(api_key)
                # Use model_dump() instead of dict() for Pydantic v2 compatibility
                return {"status": "success", "data": response.model_dump()}
            except Exception as e:
                error_message = str(e).lower()
                if "unauthorized" in error_message or ("invalid" in error_message and "key" in error_message):
                    if api_key: _blacklist_mistral_client_key(api_key)
                else:
                    if api_key: _release_mistral_client_key(api_key)
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                    console.print(f"❌ [Mistral Batch Get] API error (attempt {attempt+1}): {str(e)[:100]}... Retrying in {delay}s.", style="yellow")
                    await asyncio.sleep(delay)
                    continue
                raise
    except Exception as e:
        console.print(f"[Mistral Batch Get Processor] [bold red]Error getting batch job {request_id} for ID {job_id}: {e}[/bold red]")
        console.print_exception()
        return {"status": "error", "error_message": f"Mistral batch job retrieval failed: {e}"}

async def process_mistral_batch_job_list(request_id: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = None
    try:
        for attempt in range(MAX_RETRIES):
            api_key = None
            try:
                api_key = await _acquire_mistral_client_key()
                client = mistral_client_pool[api_key]
                
                response = await asyncio.to_thread(client.batch.jobs.list, **query_params)
                _release_mistral_client_key(api_key)
                # Use model_dump() instead of dict() for Pydantic v2 compatibility
                return {"status": "success", "data": [job.model_dump() for job in response.data]} # List of BatchJob objects
            except Exception as e:
                error_message = str(e).lower()
                if "unauthorized" in error_message or ("invalid" in error_message and "key" in error_message):
                    if api_key: _blacklist_mistral_client_key(api_key)
                else:
                    if api_key: _release_mistral_client_key(api_key)
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                    console.print(f"❌ [Mistral Batch List] API error (attempt {attempt+1}): {str(e)[:100]}... Retrying in {delay}s.", style="yellow")
                    await asyncio.sleep(delay)
                    continue
                raise
    except Exception as e:
        console.print(f"[Mistral Batch List Processor] [bold red]Error listing batch jobs {request_id}: {e}[/bold red]")
        console.print_exception()
        return {"status": "error", "error_message": f"Mistral batch job listing failed: {e}"}

async def process_mistral_batch_job_cancel(request_id: str, job_id: str) -> Dict[str, Any]:
    api_key = None
    try:
        for attempt in range(MAX_RETRIES):
            api_key = None
            try:
                api_key = await _acquire_mistral_client_key()
                client = mistral_client_pool[api_key]
                
                response = await asyncio.to_thread(client.batch.jobs.cancel, job_id=job_id)
                _release_mistral_client_key(api_key)
                # Use model_dump() instead of dict() for Pydantic v2 compatibility
                return {"status": "success", "data": response.model_dump()}
            except Exception as e:
                error_message = str(e).lower()
                if "unauthorized" in error_message or ("invalid" in error_message and "key" in error_message):
                    if api_key: _blacklist_mistral_client_key(api_key)
                else:
                    if api_key: _release_mistral_client_key(api_key)
                
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                    console.print(f"❌ [Mistral Batch Cancel] API error (attempt {attempt+1}): {str(e)[:100]}... Retrying in {delay}s.", style="yellow")
                    await asyncio.sleep(delay)
                    continue
                raise
    except Exception as e:
        console.print(f"[Mistral Batch Cancel Processor] [bold red]Error canceling batch job {request_id} for ID {job_id}: {e}[/bold red]")
        console.print_exception()
        return {"status": "error", "error_message": f"Mistral batch job cancellation failed: {e}"}

# ==============================================================================
# 8. Web Handlers and Application Setup
# ==============================================================================

async def handle_chat_completions(request: web.Request) -> web.Response:
    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body."}, status=400)

    if request_data.get("stream", False):
        return web.json_response({"error": "Streaming not supported by this unified server"}, status=400)

    model_name = request_data.get("model", "").lower()

    if model_name in GEMINI_MODEL_NAMES:
        console.print(f"➡️ Routing request to [bold purple]Gemini[/bold purple] for model: {model_name}")
        result = await process_gemini_call(request_data)
    else:
        console.print(f"➡️ Routing request to [bold blue]Mistral[/bold blue] for model: {model_name or DEFAULT_MISTRAL_MODEL}")
        result = await process_mistral_call(request_data)

    if result.get("status") == "success":
        return web.json_response(result["data"])
    else:
        error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
        return web.json_response(error_resp, status=500)

async def handle_ocr(request: web.Request) -> web.Response:
    try:
        request_data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body."}, status=400)

    model_name = request_data.get("model", "mistral-ocr-latest").lower()
    if "ocr" not in model_name:
        return web.json_response({"error": f"Invalid model for OCR. Please use a model like 'mistral-ocr-latest'. Provided: {model_name}"}, status=400)

    console.print(f"➡️ Routing request to [bold yellow]Mistral OCR[/bold yellow] for model: {model_name}")
    result = await process_mistral_ocr_call(request_data)

    if result.get("status") == "success":
        return web.json_response(result["data"])
    else:
        error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
        return web.json_response(error_resp, status=500)

async def handle_mistral_file_upload(request: web.Request) -> web.Response:
    request_id = f"req_file_upload_{int(time.time() * 1000)}"
    try:
        reader = await request.multipart()
        field = await reader.next()
        if field.name != 'file':
            return web.json_response({"error": "Missing 'file' part in multipart/form-data"}, status=400)
        
        file_name = field.filename
        file_content = await field.read()

        purpose_field = await reader.next()
        if purpose_field.name != 'purpose':
            return web.json_response({"error": "Missing 'purpose' part in multipart/form-data"}, status=400)
        purpose = (await purpose_field.read()).decode('utf-8')

        if not file_name or not file_content:
            return web.json_response({"error": "File content and name are required"}, status=400)
        if purpose not in ["batch", "assistants"]: # Mistral docs specify "batch" or "assistants"
            return web.json_response({"error": "Invalid purpose. Must be 'batch' or 'assistants'."}, status=400)

        console.print(f"⬆️ [bold blue]Mistral File Upload[/bold blue]: Uploading '{file_name}' for purpose '{purpose}'")
        result = await process_mistral_file_upload(request_id, file_name, file_content, purpose)

        if result.get("status") == "success":
            return web.json_response(result["data"])
        else:
            error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
            return web.json_response(error_resp, status=500)
    except Exception as e:
        console.print_exception()
        return web.json_response({"error": f"Error handling file upload: {e}"}, status=500)

async def handle_mistral_file_download(request: web.Request) -> web.Response:
    request_id = f"req_file_download_{int(time.time() * 1000)}"
    file_id = request.match_info.get("file_id")

    if not file_id:
        return web.json_response({"error": "File ID is required in the path."}, status=400)

    console.print(f"⬇️ [bold blue]Mistral File Download[/bold blue]: Downloading file ID '{file_id}'")
    result = await process_mistral_file_download(request_id, file_id)

    if result.get("status") == "success":
        # Determine content type based on expected file extensions, default to octet-stream
        content_type = 'application/jsonl' if file_id.endswith('.jsonl') else 'application/octet-stream'
        return web.Response(body=result["data"], content_type=content_type)
    else:
        error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
        return web.json_response(error_resp, status=500)

async def handle_mistral_batch_jobs_list_create(request: web.Request) -> web.Response:
    if request.method == "POST":
        request_id = f"req_batch_create_{int(time.time() * 1000)}"
        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON body."}, status=400)
        
        # Validate required fields for create
        required_fields = ["input_files", "model", "endpoint"]
        if not all(field in request_data for field in required_fields):
            return web.json_response({"error": f"Missing required fields for batch job creation: {', '.join(required_fields)}"}, status=400)

        console.print(f"➕ [bold blue]Mistral Batch Job[/bold blue]: Creating new batch job for model '{request_data.get('model')}'")
        result = await process_mistral_batch_job_create(request_id, request_data)
        if result.get("status") == "success":
            return web.json_response(result["data"], status=200)
        else:
            error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
            return web.json_response(error_resp, status=500)
    elif request.method == "GET":
        request_id = f"req_batch_list_{int(time.time() * 1000)}"
        query_params = dict(request.query)
        # Convert metadata[key] to metadata={"key": value} if present
        metadata_dict = {}
        for k, v in list(query_params.items()):
            if k.startswith("metadata[") and k.endswith("]"):
                meta_key = k[len("metadata["):-1]
                metadata_dict[meta_key] = v
                del query_params[k]
        if metadata_dict:
            query_params["metadata"] = metadata_dict
        
        console.print(f"📝 [bold blue]Mistral Batch Job[/bold blue]: Listing batch jobs with filters: {query_params}")
        result = await process_mistral_batch_job_list(request_id, query_params)
        if result.get("status") == "success":
            return web.json_response({"object": "list", "data": result["data"]}, status=200)
        else:
            error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
            return web.json_response(error_resp, status=500)
    else:
        return web.json_response({"error": "Method Not Allowed"}, status=405)

async def handle_mistral_batch_job_details(request: web.Request) -> web.Response:
    job_id = request.match_info.get("job_id")
    if not job_id:
        return web.json_response({"error": "Job ID is required in the path."}, status=400)

    if request.method == "GET":
        request_id = f"req_batch_get_{int(time.time() * 1000)}"
        console.print(f"🔍 [bold blue]Mistral Batch Job[/bold blue]: Getting details for job ID '{job_id}'")
        result = await process_mistral_batch_job_get(request_id, job_id)
        if result.get("status") == "success":
            return web.json_response(result["data"], status=200)
        else:
            error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
            return web.json_response(error_resp, status=500)
    elif request.method == "POST" and request.url.path.endswith(f"/{job_id}/cancel"):
        request_id = f"req_batch_cancel_{int(time.time() * 1000)}"
        console.print(f"🛑 [bold blue]Mistral Batch Job[/bold blue]: Cancelling job ID '{job_id}'")
        result = await process_mistral_batch_job_cancel(request_id, job_id)
        if result.get("status") == "success":
            return web.json_response(result["data"], status=200)
        else:
            error_resp = {"error": {"message": result.get("error_message", "Unknown server error"), "type": "api_error"}}
            return web.json_response(error_resp, status=500)
    else:
        return web.json_response({"error": "Method Not Allowed or Invalid Action"}, status=405)

async def handle_models(request: web.Request) -> web.Response:
    # Copied and merged from both sources
    return web.json_response({ "object": "list", "data": [
        {"id": "mistral-large-latest", "object": "model", "owned_by": "mistralai"},
        {"id": "mistral-medium-latest", "object": "model", "owned_by": "mistralai"},
        {"id": "mistral-small-latest", "object": "model", "owned_by": "mistralai"},
        {"id": "mistral-ocr-latest", "object": "model", "owned_by": "mistralai"},
        {"id": "pixtral-large-latest", "object": "model", "owned_by": "mistralai"},
        {"id": "gemini-1.5-pro-latest", "object": "model", "owned_by": "google"},
        {"id": "gemini-pro", "object": "model", "owned_by": "google"},
        {"id": "gpt-4o", "object": "model", "owned_by": "openai-compatible/google"},
        {"id": "gpt-4-vision-preview", "object": "model", "owned_by": "openai-compatible/google"},
    ]})

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "healthy",
        "mistral_keys_in_pool": mistral_key_queue.qsize() if mistral_key_queue else 0,
        "gemini_keys_in_pool": gemini_key_manager.get_active_key_count() if gemini_key_manager else 0,
        "metrics": metrics.get_dashboard_data()["stats"],
        "vision_support": True,
        "supported_vision_models": list(ALL_VISION_MODELS)
    })

async def handle_dashboard(request: web.Request) -> web.Response:
    dashboard_html = """
<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>🚀 Unified AI Gateway Dashboard</title><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script><style>
* { margin: 0; padding: 0; box-sizing: border-box; } body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #111827; color: white; min-height: 100vh; padding: 20px; } .container { max-width: 1400px; margin: 0 auto; }
.header { text-align: center; margin-bottom: 30px; } .header h1 { font-size: 2.5rem; margin-bottom: 10px; background: -webkit-linear-gradient(45deg, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header .subtitle { color: #d1d5db; opacity: 0.9; } .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #4ade80; margin-right: 8px; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } } .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
.card { background: #1f2937; border-radius: 15px; padding: 20px; border: 1px solid #374151; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
.card h3 { margin-bottom: 15px; font-size: 1.2rem; color: #e5e7eb; } .metric { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 0.95rem; }
.metric-label { color: #9ca3af; } .metric-value { font-weight: 600; color: #f9fafb; } .progress-bar { width: 100%; height: 8px; background: #374151; border-radius: 4px; overflow: hidden; margin: 5px 0; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #60a5fa, #3b82f6); transition: width 0.5s ease; } .active-requests { max-height: 220px; overflow-y: auto; }
.request-item { background: #374151; padding: 8px 12px; margin: 6px 0; border-radius: 8px; font-size: 0.9rem; display: flex; justify-content: space-between; align-items: center; }
.request-item.with-images { border-left: 4px solid #f59e0b; padding-left: 8px; } .chart-container { position: relative; height: 300px; } canvas { border-radius: 10px; }
.error-item { display: flex; justify-content: space-between; margin: 5px 0; } .large-number { font-size: 2rem; font-weight: bold; color: #60a5fa; }
.connection-status { position: fixed; top: 20px; right: 20px; padding: 10px; background: rgba(0,0,0,0.8); border-radius: 8px; font-size: 0.9rem; z-index: 100; }
.vision-indicator { display: inline-block; padding: 2px 6px; background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.5); border-radius: 4px; font-size: 0.7rem; font-weight: bold; color: #fbbf24; margin-left: 8px; }
.model-tag { padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 500; }
.model-gemini { background-color: rgba(139, 92, 246, 0.2); color: #c4b5fd; } .model-mistral { background-color: rgba(59, 130, 246, 0.2); color: #93c5fd; }
</style></head><body><div class="connection-status" id="connectionStatus"><span class="status-indicator"></span> Connecting...</div><div class="container"><div class="header"><h1>🚀 Unified AI Gateway</h1><p class="subtitle">Real-time monitoring for Mistral & Gemini backends</p></div>
<div class="grid"><div class="card"><h3>📊 Overview</h3> <div class="metric"><span class="metric-label">Total Requests</span><span class="metric-value large-number" id="totalRequests">0</span></div><div class="metric"><span class="metric-label">Success Rate</span><span class="metric-value" id="successRate">0%</span></div><div class="metric"><span class="metric-label">Avg Response Time</span><span class="metric-value" id="avgResponseTime">0ms</span></div><div class="metric"><span class="metric-label">Requests/min</span><span class="metric-value" id="requestsPerMin">0</span></div><div class="metric"><span class="metric-label">Uptime</span><span class="metric-value" id="uptime">0s</span></div></div>
<div class="card"><h3>🔑 API Keys</h3> <div class="metric"><span class="metric-label">Mistral Keys Active</span><span class="metric-value" id="mistralKeys">0</span></div><div class="metric"><span class="metric-label">Gemini Keys Active</span><span class="metric-value" id="geminiKeys">0</span></div></div>
<div class="card"><h3>⚡ Current Load</h3> <div class="metric"><span class="metric-label">Active Requests</span><span class="metric-value" id="activeCount">0</span></div><div class="progress-bar"><div class="progress-fill" id="loadProgress" style="width: 0%"></div></div><div class="metric"><span class="metric-label">Peak Concurrent</span><span class="metric-value" id="peakConcurrent">0</span></div><div class="metric"><span class="metric-label">Multimodal %</span><span class="metric-value" id="multimodalPercentage">0%</span></div></div>
<div class="card" style="grid-column: 1 / -1;"><h3>🔄 Active Requests</h3><div class="active-requests" id="activeRequestsList"><p style="opacity: 0.6;">No active requests</p></div></div>
<div class="card" style="grid-column: 1 / -1;"><h3>📈 Response Time (Last 50 requests)</h3><div class="chart-container"><canvas id="responseTimeChart"></canvas></div></div>
<div class="card" style="grid-column: 1 / 2;"><h3>🤖 Model Usage</h3><div id="modelUsage"><p style="opacity: 0.6;">No usage recorded</p></div></div>
<div class="card" style="grid-column: 2 / -1;"><h3>❌ Error Breakdown</h3><div id="errorBreakdown"><p style="opacity: 0.6;">No errors recorded</p></div></div></div></div>
<script>
let ws,reconnectAttempts=0,charts={};const GEMINI_MODELS=['gemini','gpt-4o','gpt-4-vision-preview'];
function updateConnectionStatus(c){const s=document.getElementById('connectionStatus');s.innerHTML=`<span class="status-indicator" style="background:${c?'#4ade80':'#ef4444'}"></span> ${c?'Connected':'Disconnected'}`;s.style.background=c?'rgba(34,197,94,0.8)':'rgba(239,68,68,0.8)';if(c)reconnectAttempts=0}
function connectWebSocket(){const p=window.location.protocol==='https:'?'wss:':'ws:',w=`${p}//${window.location.hostname}:${window.location.port}/ws`;ws=new WebSocket(w);ws.onopen=()=>{updateConnectionStatus(true)};ws.onmessage=e=>{updateDashboard(JSON.parse(e.data))};ws.onclose=()=>{updateConnectionStatus(false);const d=Math.min(1e3*Math.pow(2,reconnectAttempts),3e4);setTimeout(()=>{reconnectAttempts++;connectWebSocket()},d)};ws.onerror=e=>{console.error('WebSocket error:',e)}}
function updateDashboard(d){const s=d.stats;document.getElementById('totalRequests').textContent=s.total_requests.toLocaleString();document.getElementById('successRate').textContent=s.success_rate.toFixed(1)+'%';document.getElementById('avgResponseTime').textContent=Math.round(s.avg_response_time*1e3)+'ms';document.getElementById('requestsPerMin').textContent=s.requests_per_minute.toFixed(1);document.getElementById('uptime').textContent=s.uptime_formatted;document.getElementById('mistralKeys').textContent=s.mistral_keys_available;document.getElementById('geminiKeys').textContent=s.gemini_keys_available;document.getElementById('activeCount').textContent=s.active_count;document.getElementById('loadProgress').style.width=s.current_load+'%';document.getElementById('peakConcurrent').textContent=s.peak_concurrent;document.getElementById('multimodalPercentage').textContent=s.multimodal_percentage.toFixed(1)+'%';
const a=document.getElementById('activeRequestsList');a.innerHTML=d.active_requests.length===0?'<p style="opacity: 0.6;">No active requests</p>':d.active_requests.map(r=>{const i=r.has_images?'<span class="vision-indicator">IMAGE</span>':'';const m=GEMINI_MODELS.some(gm=>r.model.includes(gm))||r.model.includes('ocr');const t=`<span class="model-tag ${m?'model-gemini':'model-mistral'}">${r.model.includes('ocr')?'Mistral OCR':(m?'Gemini':'Mistral')}</span>`;return `<div class="request-item ${r.has_images?'with-images':''}"><div style="flex-grow:1;"><strong>${r.id}</strong><br><small>${r.model} | ${r.duration.toFixed(1)}s elapsed</small></div>${t}${i}</div>`}).join('');
updateResponseTimeChart(d.recent_performance);const mu=document.getElementById('modelUsage');mu.innerHTML=Object.keys(d.model_usage).length===0?'<p style="opacity: 0.6;">No usage recorded</p>':Object.entries(d.model_usage).sort(([,a],[,b])=>b-a).map(([m,c])=>`<div class="metric"><span class="metric-label">${m}</span><span class="metric-value">${c.toLocaleString()}</span></div>`).join('');
const b=document.getElementById('errorBreakdown');b.innerHTML=Object.keys(d.error_breakdown).length===0?'<p style="opacity: 0.6;">No errors recorded</p>':Object.entries(d.error_breakdown).map(([e,c])=>`<div class="error-item"><span>${e}</span><span>${c}</span></div>`).join('')}
function initResponseTimeChart(){const c=document.getElementById('responseTimeChart').getContext('2d');charts.responseTime=new Chart(c,{type:'scatter',data:{datasets:[{label:'Text Request',data:[],backgroundColor:'rgba(96,165,250,0.8)'},{label:'Image Request',data:[],backgroundColor:'rgba(251,191,36,0.8)'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,labels:{color:'white'}}},scales:{x:{type:'linear',display:false},y:{beginAtZero:true,grid:{color:'rgba(255,255,255,0.1)'},ticks:{color:'rgba(255,255,255,0.8)',callback:v=>v+'ms'},title:{display:true,text:'Response Time (ms)',color:'white'}}}}})}
function updateResponseTimeChart(p){if(!charts.responseTime)return;const c=charts.responseTime,t=[],i=[];p.forEach((d,a)=>{const o={x:a,y:d.response_time};if(d.has_images){i.push(o)}else{t.push(o)}});c.data.datasets[0].data=t;c.data.datasets[1].data=i;c.update('none')}
document.addEventListener('DOMContentLoaded',()=>{initResponseTimeChart();connectWebSocket()});
</script></body></html>
    """
    return web.Response(text=dashboard_html, content_type='text/html')

async def handle_dashboard_websocket(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    metrics.websocket_clients.add(ws)
    try:
        await ws.send_str(json.dumps(metrics.get_dashboard_data()))
        async for msg in ws:
            if msg.type == WSMsgType.ERROR: break
    finally:
        metrics.websocket_clients.discard(ws)
    return ws

async def broadcast_metrics_updates():
    while True:
        await asyncio.sleep(1.0)
        if not metrics.websocket_clients: continue
        try:
            message = json.dumps(metrics.get_dashboard_data())
            disconnected = {ws for ws in metrics.websocket_clients if ws.closed}
            metrics.websocket_clients -= disconnected
            await asyncio.gather(*(ws.send_str(message) for ws in metrics.websocket_clients), return_exceptions=False)
        except Exception: pass

async def on_shutdown(app: web.Application):
    console.print("--- Server shutting down ---", style="bold yellow")
    if 'gemini_key_manager' in app and app['gemini_key_manager']:
        app['gemini_key_manager'].shutdown()
    console.print("--- Shutdown complete ---", style="bold green")

async def init_app() -> web.Application:
    global request_semaphore, gemini_key_manager, gemini_generator, mistral_key_queue

    console.print("🚀 [bold]Starting Unified Gemini-Mistral API Gateway...[/bold]")

    # --- Initialize Mistral Components ---
    console.print("🔄 [Mistral] Initializing components...", style="blue")
    mistral_keys = load_mistral_keys(MISTRAL_SHEET_URL)
    if not mistral_keys: console.print("⚠️ [Mistral] No API keys loaded. Mistral backend will be unavailable.", style="yellow")
    mistral_key_queue = asyncio.Queue()
    for key in mistral_keys: await mistral_key_queue.put(key)
    init_mistral_client_pool(mistral_keys)
    console.print(f"✅ [Mistral] Components initialized with {len(mistral_keys)} keys.", style="green")

    # --- Initialize Gemini Components ---
    console.print("🔄 [Gemini] Initializing components...", style="purple")
    if genai:
        try:
            gemini_key_manager = ApiKeyManager(sheet_url=GEMINI_SHEET_URL)
            gemini_generator = GeneratorGemini(api_key_manager=gemini_key_manager)
            console.print("✅ [Gemini] Components initialized successfully.", style="green")
        except Exception as e:
            console.print(f"❌ [bold red]CRITICAL: Failed to initialize Gemini components: {e}[/bold red]")
            gemini_key_manager, gemini_generator = None, None
    else:
        console.print("⚠️ [Gemini] 'google-generativeai' not installed. Gemini backend is disabled.", style="yellow")
        gemini_key_manager, gemini_generator = None, None

    request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    app = web.Application(client_max_size=1024**2 * 100)
    
    # Chat Completions and OCR Endpoints
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/ocr", handle_ocr)
    
    # Mistral Batch Inference and File Management Endpoints
    app.router.add_post("/v1/files", handle_mistral_file_upload)
    app.router.add_get("/v1/files/{file_id}/content", handle_mistral_file_download)
    
    app.router.add_route("GET", "/v1/batch/jobs", handle_mistral_batch_jobs_list_create)
    app.router.add_route("POST", "/v1/batch/jobs", handle_mistral_batch_jobs_list_create)
    app.router.add_route("GET", "/v1/batch/jobs/{job_id}", handle_mistral_batch_job_details)
    app.router.add_route("POST", "/v1/batch/jobs/{job_id}/cancel", handle_mistral_batch_job_details)

    # General Endpoints
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/", lambda r: web.HTTPFound('/dashboard'))
    app.router.add_get("/dashboard", handle_dashboard)
    app.router.add_get("/ws", handle_dashboard_websocket)
    
    app.on_shutdown.append(on_shutdown)
    return app

async def main():
    if os.name != 'nt':
        try:
            import uvloop
            uvloop.install()
            console.print("🏃 Using [cyan]uvloop[/cyan] for high-performance event loop.")
        except ImportError: pass
    
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    
    metrics_task = asyncio.create_task(broadcast_metrics_updates())
    
    try:
        await site.start()
        console.print(f"✅ [bold green]Server is running at http://{HOST}:{PORT}[/bold green]")
        console.print("   Press Ctrl+C to stop.")
        await asyncio.Event().wait()
    except KeyboardInterrupt: pass
    finally:
        console.print("\n🛑 Stopping server...", style="yellow")
        metrics_task.cancel()
        await runner.cleanup()
        console.print("👋 Server shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"❌ [bold red]Failed to start server: {e}[/bold red]")