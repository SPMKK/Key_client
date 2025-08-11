from openai import OpenAI
import re 
# from gemini_mistral_server import MISTRAL_SHEET_URL, GEMINI_SHEET_URL
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import base64
import mimetypes
from io import BytesIO
from typing_extensions import TypedDict, Literal    
# Define a type for the message structure used in the API
from typing import Union, List, Dict, Any, Optional, Tuple
import json
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[Dict[str, Any]]]
import contextlib
import m3u8
from pydub import AudioSegment # Add this new import
import subprocess
import tempfile
import os
import math
import time
import asyncio
import concurrent.futures
import functools
import mimetypes
import unicodedata
from pydub.silence import split_on_silence
# from io import BytesIO
def remove_vietnamese_tones(text):
    # Normalize Unicode (NFD = tách ký tự và dấu)
    text = unicodedata.normalize('NFD', text)
    # Loại bỏ các ký tự dạng dấu (combining diacritical marks)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    # Convert về dạng chuẩn NFC nếu cần (tùy use-case)
    return text
class Generator:
    """
    An optimized, universal client for the Unified AI Gateway server.
    Supports both text and multimodal (text and image) generation, intelligently
    leveraging the server's backend routing for Mistral and Gemini models.
    """
    def __init__(self,
                 base_url: str = "http://localhost:9501",
                 api_key: str = "dummy", # The gateway manages keys, so this can be a placeholder
                 model_name: str = "mistral-medium-latest",
                 temperature: float = 0.7,
                 max_new_tokens: int = 4096,
                 timeout: int = 120):
        """
        Initialize the Generator for the Unified AI Gateway.

        Args:
            base_url: URL of your AI Gateway server.
            api_key: API key (can be a dummy value for the gateway).
            model_name: Default model name to use for requests.
            temperature: Default temperature for generation.
            max_new_tokens: Default maximum number of new tokens to generate.
            timeout: Request timeout in seconds (increased for vision models).
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

        # Initialize OpenAI client pointed at your server's endpoint
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key=api_key,
            timeout=timeout
        )

        print(f"✅ Generator Initialized: Connected to Unified AI Gateway at {self.base_url}")
        self._test_connection()

    def _test_connection(self) -> None:
        """Tests the connection to the server and displays its capabilities."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"🏥 Server Health: {health_data.get('status', 'unknown').capitalize()}")

                if health_data.get('vision_support'):
                    print("📷 Vision Support: Enabled")
                    vision_models = health_data.get('supported_vision_models', [])
                    if vision_models:
                        print(f"🎯 Supported Vision Models: {', '.join(vision_models)}")
                else:
                    print("📷 Vision Support: Disabled")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not fetch server health info: {e}")
            print("🔄 Proceeding with basic configuration, but server may be offline.")

    def get_available_models(self) -> List[str]:
        """Gets the list of available models from the server."""
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            print(f"⚠️ Could not fetch models from the gateway: {e}")
            return []

    def _encode_image_from_path(self, image_path: str) -> str:
        """Loads an image file, encodes it to Base64, and formats it as a data URI."""
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found at {image_path}")

            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith('image'):
                ext_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'}
                mime_type = ext_map.get(path.suffix.lower())
                if not mime_type:
                    raise ValueError(f"Unsupported or unknown image format for file: {path.name}")

            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            raise IOError(f"Error processing image file {image_path}: {e}") from e

    def _encode_image_from_pil(self, image: Image.Image) -> str:
        """Encodes a PIL.Image object to a Base64 data URI."""
        try:
            buffered = BytesIO()
            # Save as PNG to preserve quality; it's a safe, widely supported format.
            image.save(buffered, format="PNG")
            base64_bytes = base64.b64encode(buffered.getvalue())
            return f"data:image/png;base64,{base64_bytes.decode('utf-8')}"
        except Exception as e:
            raise IOError(f"Error processing PIL image object: {e}") from e

    def _prepare_api_messages(self,
                            prompt: Optional[str],
                            messages: Optional[List[Message]],
                            images: Optional[List[Union[str, Image.Image]]]) -> Tuple[List[Message], List[Message], bool]:
        """Prepares messages for the API server, handling text and various image input types."""
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        full_history = list(messages) if messages else []

        if not full_history:
            full_history.append({"role": "user", "content": prompt or ""})

        if not images:
            return full_history, full_history, False

        # Images can only be attached to the last user message
        last_message = full_history[-1]
        if last_message["role"] != "user":
            raise ValueError("Images can only be added to the most recent 'user' message in the history.")

        # Ensure content is a string before modification
        text_content = last_message.get("content", "")
        if not isinstance(text_content, str):
            raise TypeError("The last message content must be a string when providing new images.")

        has_images = False
        final_user_content = [{"type": "text", "text": text_content}]

        for i, img_input in enumerate(images):
            try:
                if isinstance(img_input, str):
                    base64_image_uri = self._encode_image_from_path(img_input)
                elif isinstance(img_input, Image.Image):
                    base64_image_uri = self._encode_image_from_pil(img_input)
                else:
                    raise TypeError(f"Image input must be a file path (str) or a PIL.Image.Image object, but got {type(img_input)}")

                final_user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_image_uri}
                })
                has_images = True
            except (IOError, FileNotFoundError, ValueError, TypeError) as e:
                error_msg = f"[Error processing image {i+1}: {e}]"
                # Prepend the error to the text content for the LLM to see
                final_user_content[0]["text"] = f"{error_msg}\n{final_user_content[0]['text']}"
                print(f"⚠️  {error_msg}")

        # Replace the last message with the new multimodal content
        api_messages = full_history[:-1]
        api_messages.append({"role": "user", "content": final_user_content})

        return api_messages, full_history, has_images

    def _handle_api_error(self, error: Exception) -> str:
        """Provides user-friendly interpretations of potential API errors."""
        error_str = str(error)
        if "Connection refused" in error_str:
            return f"Connection Failed. Is the AI Gateway server running at {self.base_url}?"
        if "All Gemini API keys have been exhausted" in error_str:
            return "Server reported that all Gemini keys are exhausted or have failed. Please check the server logs."
        if "Mistral request failed after" in error_str:
            return "Server reported that all retries for Mistral failed. The Mistral API might be down or all keys are invalid."
        return f"An unhandled error occurred: {error_str}"

    def generate(self,
                 prompt: Optional[str] = None,
                 messages: Optional[List[Message]] = None,
                 images: Optional[List[Union[str, Image.Image]]] = None,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_new_tokens: Optional[int] = None,
                 **kwargs) -> Tuple[str, List[Message]]:
        """
        Generates a response using the Unified AI Gateway.

        The gateway server automatically:
        - Routes requests to Mistral or Gemini based on the model name.
        - Manages API key rotation, rate limiting, and load balancing.
        - Provides real-time metrics and monitoring.

        Args:
            prompt: A single string prompt. Used if 'messages' is not provided.
            messages: A list of message dictionaries (OpenAI format).
            images: A list of image inputs. Items can be file paths (str) or PIL.Image.Image objects.
            model_name: The model to use (e.g., "gpt-4o", "mistral-medium-latest").
            temperature: The temperature for this request.
            max_new_tokens: The max tokens for this request.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A tuple containing:
            - The generated text response (str).
            - The complete conversation history including the new response (list).
        """
        try:
            api_messages, conversation_history, has_images = self._prepare_api_messages(
                prompt, messages, images
            )

            selected_model = model_name if model_name is not None else self.model_name

            if has_images:
                num_images = len([part for part in api_messages[-1]['content'] if part.get('type') == 'image_url'])
                print(f"📷 Sending multimodal request to model '{selected_model}' with {num_images} image(s)")

            api_params = {
                "model": selected_model,
                "messages": api_messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
                **kwargs
            }

            response = self.client.chat.completions.create(**api_params)
            result = response.choices[0].message.content or ""

            if response.model and response.model != selected_model:
                print(f"🔄 Server auto-routed request to model: {response.model} (requested: {selected_model})")

        except Exception as e:
            error_message = self._handle_api_error(e)
            print(f"❌ Generation failed: {error_message}")
            raise RuntimeError(error_message) from e

        conversation_history.append({"role": "assistant", "content": result})
        return result, conversation_history
    
    def _handle_m3u8_playlist(self, file_path: str) -> bytes:
        """
        Parses and processes an M3U8 playlist, correctly handling both
        web URLs and local file paths for media segments.
        """
        # print(f"📄 Playlist file detected. Parsing and processing media from '{Path(file_path).name}'...")
        try:
            # The base path of the m3u8 file is used to resolve relative segment paths.
            base_path = Path(file_path).parent
            playlist = m3u8.load(file_path)
            
            if playlist.is_variant:
                # print("ℹ️ Master playlist found. Selecting first available stream.")
                # The URI in a master playlist can be another m3u8 file
                sub_playlist_uri = playlist.playlists[0].uri
                # Resolve its path relative to the master playlist's path
                playlist = m3u8.load(str(base_path / sub_playlist_uri))

            media_segments = []
            total_segments = len(playlist.segments)
            # print(f"⚙️ Processing {total_segments} media segments...")

            for i, segment in enumerate(playlist.segments):
                segment_uri = segment.uri
                
                # --- CORRECTED LOGIC: Differentiate between URL and local path ---
                if segment_uri.startswith(('http://', 'https://')):
                    # It's a full web URL, download it.
                    response = requests.get(segment_uri, timeout=30)
                    response.raise_for_status()
                    media_segments.append(response.content)
                else:
                    # It's a relative local file path. Construct the full path and read it.
                    local_segment_path = base_path / segment_uri
                    if not local_segment_path.exists():
                        raise FileNotFoundError(f"Media segment file not found: {local_segment_path}")
                    
                    with open(local_segment_path, 'rb') as f_segment:
                        media_segments.append(f_segment.read())
                # --- END OF CORRECTION ---

                print(f"   Processed segment {i+1}/{total_segments}", end='\r')

            # print("\n✅ Processing complete. Stitching segments together.")
            return b"".join(media_segments)

        except Exception as e:
            raise IOError(f"Failed to process M3U8 playlist '{file_path}': {e}") from e

    def chunk_audio_for_gemini(
        self,
        audio_path,
        max_chunk_mb=18,          # Keep a bit below 20 MB inline limit
        min_chunk_sec=1.0,        # Skip anything shorter than 1 sec
        min_loudness_dbfs=-50     # Skip chunks quieter than this
    ):
        """
        Split audio into safe chunks for Gemini API.
        Returns a list of dicts: [{"bytes": ..., "name": "chunk_x.mp3"}]
        """
        audio = AudioSegment.from_file(audio_path)
        processed_chunks = []

        # 1. Split on silence
        speech_segments = split_on_silence(
            audio,
            min_silence_len=700,                   # 0.7 seconds
            silence_thresh=audio.dBFS - 14,        # 14 dB below avg
            keep_silence=300                       # 0.3 sec padding
        )

        if not speech_segments:
            print("[INFO] No speech detected.")
            return []

        print(f"[INFO] Found {len(speech_segments)} raw segments.")

        # 2. Combine segments into larger chunks under size limit
        current_chunk = AudioSegment.empty()
        chunk_index = 1

        def finalize_chunk(ch):
            """Export chunk to bytes and check size/quality."""
            buf = BytesIO()
            ch.export(buf, format="mp3", bitrate="128k")
            size_mb = len(buf.getvalue()) / (1024 * 1024)

            # Skip empty, too short, or too quiet
            if size_mb < 0.05:  # 50 KB
                return None
            if len(ch) < min_chunk_sec * 1000:
                return None
            if ch.dBFS < min_loudness_dbfs:
                return None
            if size_mb > max_chunk_mb:
                print(f"[WARN] A single chunk exceeded {max_chunk_mb} MB after encoding.")
                return None

            return {
                "bytes": buf.getvalue(),
                "name": f"chunk_{chunk_index}.mp3"
            }

        for seg in speech_segments:
            # Test if adding seg would exceed limit
            test_chunk = current_chunk + seg
            buf = BytesIO()
            test_chunk.export(buf, format="mp3", bitrate="128k")
            size_mb = len(buf.getvalue()) / (1024 * 1024)

            if size_mb <= max_chunk_mb:
                current_chunk = test_chunk
            else:
                safe = finalize_chunk(current_chunk)
                if safe:
                    processed_chunks.append(safe)
                    chunk_index += 1
                current_chunk = seg

        # Add the last chunk
        safe = finalize_chunk(current_chunk)
        if safe:
            processed_chunks.append(safe)

        print(f"[INFO] Final: {len(processed_chunks)} safe chunks for Gemini.")
        return processed_chunks
    def _transcribe_chunk(self,
                          # NO session_key
                          file_bytes: bytes,
                          file_name: str,
                          model: str,
                          prompt: Optional[str],
                          language: Optional[str]) -> str:
        """A private helper to transcribe a single, stateless chunk."""
        transcription_url = f"{self.base_url}/v1/audio/transcriptions"
        
        # Simple payload, no session key
        data_payload = {'model': model}
        if prompt: data_payload['prompt'] = prompt
        if language: data_payload['language'] = language
        
        files_payload = {'file': (file_name, file_bytes)}
        
        try:
            response = requests.post(
                transcription_url,
                files=files_payload,
                data=data_payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            if "text" not in result:
                raise Exception(f"Server returned an unexpected response for chunk {file_name}: {result}")
            print(f"✅ Transcribed chunk '{file_name}' successfully.")
            return result["text"]
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe chunk {file_name}") from e

    async def transcribe(self,
                         file: Union[str, bytes],
                         model: str = "gemini-1.5-pro-latest",
                         file_name: Optional[str] = "media.dat",
                         prompt: Optional[str] = None,
                         language: Optional[str] = None,
                         max_chunk_mb: int = 15,
                         max_concurrent_chunks: int = 10) -> str: # Concurrency limit
        """
        Transcribes an audio or video file concurrently by splitting large files
        into smaller chunks and processing them in parallel.
        
        NOTE: This is now an ASYNC method and must be called with 'await'.
        """
        print(f"🎤 Preparing concurrent transcription request for model '{model}'...")
        
        # We need the asyncio loop for running sync code in threads
        loop = asyncio.get_running_loop()
        temp_dir = tempfile.mkdtemp()
        try:
                        # --- STAGE 1: BEGIN TRANSCRIPTION SESSION AND GET A STICKY KEY ---

            input_path_str = None
            if isinstance(file, str):
                if not Path(file).exists(): raise FileNotFoundError(f"Input file not found: {file}")
                input_path_str = file
            elif isinstance(file, bytes):
                input_path = Path(temp_dir) / file_name
                with open(input_path, 'wb') as f: f.write(file)
                input_path_str = str(input_path)
            else:
                raise TypeError(f"File input must be a file path (str) or bytes, but got {type(file)}")

            standardized_audio_path = input_path_str
            # --- Stage 3: Prepare chunks for transcription ---
            # processed_size_bytes = os.path.getsize(standardized_audio_path)
            processed_size_mb = os.path.getsize(standardized_audio_path) / (1024 * 1024)
            print(f"✅ Standardized audio size: {processed_size_mb:.2f} MB")

            chunks_to_process = []
            if processed_size_mb <= max_chunk_mb:
                # File is small enough, no chunking needed.
                with open(standardized_audio_path, 'rb') as f:
                    file_content = f.read()
                chunks_to_process.append({ "bytes": file_content, "name": "audio.mp3", "prompt": prompt })
            else:
                print(f"⚠️ File exceeds {max_chunk_mb} MB. Splitting on silence + recombining...")
                chunks = self.chunk_audio_for_gemini(
                    input_path_str,
                    max_chunk_mb=max_chunk_mb,
                    min_chunk_sec=1.0,
                    min_loudness_dbfs=-50.0,
                )
                # Fallback prompt for subsequent chunks
                chunks_to_process = []
                for i, ch in enumerate(chunks):
                    chunks_to_process.append({
                        "bytes": ch["bytes"],
                        "name": ch["name"],
                        "prompt": ""
                    })

            if not chunks_to_process:
                print("[INFO] No valid chunks to transcribe.")
                return ""
            start_time=time.time()
            # --- Stage 4: Transcribe all chunks concurrently ---
            full_transcript_parts = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_chunks) as executor:
                loop = asyncio.get_running_loop()
                tasks = []
                for i, chunk_data in enumerate(chunks_to_process):
                    # The session_key must be passed to the helper
                    task = loop.run_in_executor(
                        executor, self._transcribe_chunk,
                        # session_key, # This was missing in the previous version
                        chunk_data["bytes"], chunk_data["name"], model,
                        chunk_data["prompt"], language
                    )
                    tasks.append(task)
                
                full_transcript_parts = await asyncio.gather(*tasks)
                end_time = time.time()
                print(f"✅ All parts transcribed in {end_time - start_time:.2f} seconds.")
                transcripted_text = " ".join(full_transcript_parts)
                transcripted_text = remove_vietnamese_tones(transcripted_text)
            return transcripted_text
        except Exception as e:
            # Handle all exceptions from the process
            print(f"❌ An error occurred during the transcription process.")
            if isinstance(e, RuntimeError):
                print(f"   Details: {e}")
            else:
                print(f"   Details: {self._handle_api_error(e)}")
            raise

        finally:
            # The rest of the cleanup happens immediately without waiting for the POST to finish
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
            print("🧹 Cleaned up temporary files.")
        #OCR mistral
    def ocr(self,
            image: Union[str, Image.Image],
            model_name: str = "mistral-ocr-latest",
            include_image_base64: bool = False,
            **kwargs) -> Dict[str, Any]:
        """
        Performs Optical Character Recognition (OCR) on an image using Mistral's OCR models.

        This method sends a request to the dedicated `/v1/ocr` endpoint on the Unified AI Gateway.

        Args:
            image: The image to process. Can be a file path (str) or a PIL.Image.Image object.
            model_name: The OCR model to use (e.g., "mistral-ocr-latest").
            include_image_base64: Whether to include the base64-encoded image in the response.
            **kwargs: Additional parameters to pass to the API.

        Returns:
            A dictionary containing the structured OCR response from the server.

        Raises:
            RuntimeError: If the OCR request fails after all retries on the server side
                          or if there is a connection issue.
        """
        print(f"📄 Sending OCR request for model '{model_name}'...")
        try:
            if isinstance(image, str):
                base64_image_uri = self._encode_image_from_path(image)
            elif isinstance(image, Image.Image):
                base64_image_uri = self._encode_image_from_pil(image)
            else:
                raise TypeError(f"Image input must be a file path (str) or a PIL.Image.Image object, but got {type(image)}")

            ocr_url = f"{self.base_url}/v1/ocr"
            
            payload = {
                "model": model_name,
                "image": base64_image_uri,
                "include_image_base64": include_image_base64,
                **kwargs
            }

            response = requests.post(ocr_url, json=payload, timeout=self.timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # The server might return 200 OK but with an error message in the JSON
            if "error" in result and result.get("status") == "error":
                 error_message = result.get("error_message", "Unknown OCR error from server")
                 raise Exception(error_message)

            return result

        except requests.exceptions.HTTPError as http_err:
            try:
                # Try to parse the JSON error from the response body
                error_details = http_err.response.json()
                error_message = error_details.get("error", {}).get("message", http_err.response.text)
            except json.JSONDecodeError:
                error_message = str(http_err)
            
            print(f"❌ OCR failed: {error_message}")
            raise RuntimeError(f"OCR request failed: {error_message}") from http_err
        
        except Exception as e:
            error_message = self._handle_api_error(e)
            print(f"❌ OCR failed: {error_message}")
            raise RuntimeError(error_message) from e

    def get_server_metrics(self) -> Optional[Dict]:
        """Gets real-time metrics from the gateway's dashboard."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json().get('metrics', {})
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not fetch server metrics: {e}")
        return None

    def print_server_status(self) -> None:
        """Prints a comprehensive status report from the AI Gateway."""
        metrics = self.get_server_metrics()
        if not metrics:
            print("\n❌ Could not retrieve server status. Is the server online?")
            return

        print("\n" + "="*50)
        print("🚀 UNIFIED AI GATEWAY STATUS")
        print("="*50)
        print(f"📊 Total Requests: {metrics.get('total_requests', 0):,}")
        print(f"✅ Success Rate:   {metrics.get('success_rate', 0):.1f}%")
        print(f"⚡ Active Requests:  {metrics.get('active_count', 0)}")
        print(f"🔑 Active Keys - Mistral: {metrics.get('mistral_keys_available', 'N/A')} | Gemini: {metrics.get('gemini_keys_available', 'N/A')}")
        print(f"⏱️ Avg Response:   {metrics.get('avg_response_time', 0)*1000:.0f} ms")
        print(f"📈 Throughput:     {metrics.get('requests_per_minute', 0):.1f} req/min")
        print(f"⏰ Uptime:         {metrics.get('uptime_formatted', 'Unknown')}")
        print("="*50)

# Message = Dict[str, Union[str, List[Dict[str, str]]]]
