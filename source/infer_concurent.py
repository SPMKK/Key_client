import os
import sys
import numpy as np
import tensorflow as tf
import torch
import clip
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
from PIL import Image
import shutil
import json
from typing import List, Tuple
import RAM as RAM
# import easyocr
import re
# import google.generativeai as genai
from PIL import Image
import time
# New imports for concurrency
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading
from transformers import SiglipProcessor, SiglipModel
from gemini_mistral_server import ApiKeyManager
from Generator import Generator


# (All class definitions like TransNetV2 and other helper methods remain the same)
# ... [TransNetV2 class and other methods from previous answer go here] ...
# NOTE: To keep the answer focused, I am omitting the unchanged code.
# Assume the TransNetV2 class and helper methods from the previous answer are present.
OCR_BATCH_SIZE = 8
MISTRAL_SHEET_URL = os.getenv("MISTRAL_SHEET_URL", "https://docs.google.com/spreadsheets/d/1NAlj7OiD9apH3U47RLJK0en1wLSW78X5zqmf6NmVUA4/export?format=csv&gid=0")
GEMINI_SHEET_URL = os.getenv("GEMINI_SHEET_URL", "https://docs.google.com/spreadsheets/d/1gqlLToS3OXPA-CvfgXRnZ1A6n32eXMTkXz4ghqZxe2I/export?format=csv&gid=0")

API_KEY = ApiKeyManager(
    sheet_url=GEMINI_SHEET_URL
).get_active_key_count()
#prompt
OCR_PROMPT = '''
You are an OCR agent specialized in extracting text from **news-related images** in both **Vietnamese** and **English**.

Your task is to:

- Accurately detect and extract **all visible text** from each input image, including:
  - 📰 **Main headline / title**
  - 📝 **Subheadings / captions**
  - 📄 **Body text or any other readable content**
- Preserve the **natural reading order** (top to bottom, left to right).
- Recognize both **Vietnamese (without diacritics)** and **English** text.
- **Do not translate** the text—just extract it **verbatim** as it appears.
- If an image contains **no readable text**, return an empty string for that frame.
- Input is a **batch of images**, typically extracted from a video (e.g., keyframes).
- Output the extracted texts in a **structured format**, preserving the input order.
- Extract the channel name and time if available, but focus primarily on the main text content.
---

### 📤 Input
A list of image frames, ordered as they appear in the video:
```python
["frame_0001.jpg", "frame_0002.jpg", ..., "frame_N.jpg"]


**Return answer in this json format, no additional word:**
{
    "extracted_texts": [
        "Text from frame_0001",
        "Text from frame_0002",
        ...
        "Text from frame_N"
    ]
}

'''

class TransNetV2:
    """
    TransNetV2 model for shot boundary detection.
    This class detects scene transitions in videos.
    """
    def __init__(self, model_dir=None):
        model_dir = "transnetv2-weights"
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "transnetv2-weights/")
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"[TransNetV2] ERROR: {model_dir} is not a directory.")
            else:
                print(f"[TransNetV2] Using weights from {model_dir}.")
        # Configure TensorFlow to limit GPU memory usage
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # try:
            #     # Limit TensorFlow to use only 30% of GPU memory
            #     for gpu in gpus:
            #         tf.config.experimental.set_memory_growth(gpu, True)
            #     print("[TransNetV2] TensorFlow GPU memory growth enabled")
            # except RuntimeError as e:
            #     print(f"[TransNetV2] Error setting GPU memory growth: {e}")
                
                # --- OPTION 2: SET A HARD VRAM LIMIT (UNCOMMENT TO USE) ---
                # Use this if you need to enforce a strict memory budget for TransNetV2.
                # For example, limit it to 2048 MB (2 GB).
            memory_limit_mb = 3072
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
            print(f"[TransNetV2] TensorFlow VRAM usage limited to {memory_limit_mb} MB.")

        self._input_size = (27, 48, 3)
        try:
            self._model = tf.saved_model.load(model_dir)
        except OSError as exc:
            raise IOError(f"[TransNetV2] It seems that files in {model_dir} are corrupted or missing. "
                         f"Re-download them manually and retry. For more info, see: "
                         f"https://github.com/soCzech/TransNetV2/issues/1#issuecomment-647357796") from exc

    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self._model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    def predict_frames(self, frames: np.ndarray):
        assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
            "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []
        for inp in input_iterator():
            single_frame_pred, all_frames_pred = self.predict_raw(inp)
            predictions.append((single_frame_pred.numpy()[0, 25:75, 0],
                              all_frames_pred.numpy()[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]

    def predict_video(self, video_fn: str):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError("For `predict_video` function `ffmpeg` needs to be installed...")

        print("[TransNetV2] Extracting frames from {}".format(video_fn))
        video_stream, err = ffmpeg.input(video_fn).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return (video, *self.predict_frames(video))

    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.3):
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)


class VideoKeyframeExtractor:
    def __init__(self, transnet_weights=None, output_dir="keyframes", 
                 sample_rate=1, max_frames_per_shot=100, model_path="google/siglip2-large-patch16-512"):
        self.transnet = TransNetV2(transnet_weights)
        self.model =  SiglipModel.from_pretrained(model_path, device_map="auto", attn_implementation="sdpa", torch_dtype=torch.float16).eval()
        self.preprocess = SiglipProcessor.from_pretrained(model_path)
        self.ram = RAM.load_tag_model()
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.max_frames_per_shot = max_frames_per_shot
        self.device = "cuda"
        print(f"[CLIP] Using device: {self.device}")
        os.makedirs(output_dir, exist_ok=True)
        # Thread-safe counter for frame filenames
        self._frame_counter = 1
        self._counter_lock = threading.Lock()
        # THÊM KHÓA GPU NÀY:
        self._gpu_lock = threading.Lock()
    def get_video_fps(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def extract_video_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[int]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        frames, frame_indices, frame_idx = [], [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % self.sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_indices.append(frame_idx)
            frame_idx += 1
        cap.release()
        return frames, frame_indices
        
    def _get_next_frame_count(self):
        with self._counter_lock:
            count = self._frame_counter
            self._frame_counter += 1
            return count

    def extract_clip_features(self, frames: List[np.ndarray], shot_id: int = None) -> np.ndarray:
        features = []
        batch_size = 16
        progress_prefix = f"[CLIP][Shot {shot_id}]" if shot_id is not None else "[CLIP]"

        # Đảm bảo chỉ MỘT luồng truy cập GPU tại một thời điểm
        with self._gpu_lock: # <-- THÊM DÒNG NÀY
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Chuyển đổi và gửi đến GPU
                inputs = self.preprocess(images=batch_frames, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    batch_features_raw = self.model.get_image_features(inputs)
                    batch_features /= batch_features_raw / batch_features_raw.norm(p=2, dim=-1, keepdim=True)

                features.append(batch_features.cpu().numpy())
                
                # Giải phóng bộ nhớ GPU sau mỗi batch nếu có thể

                print(f"\r{progress_prefix} Processing frames {i + len(batch_frames)}/{len(frames)}", end="")
            
        print(f"\r{progress_prefix} Feature extraction complete.                ")
        return np.vstack(features)

    def adaptive_clustering(self, features: np.ndarray) -> List[int]:
        n_samples = features.shape[0]
        if n_samples <= 1: return [0] if n_samples == 1 else []
        if n_samples <= 3: return list(range(n_samples))

        k_max = min(int(np.sqrt(n_samples)), 10)
        best_score, best_k, best_labels = -1, 2, None

        for k in range(2, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(features, labels)
                if score > best_score:
                    best_score, best_k, best_labels = score, k, labels
                    best_centers = kmeans.cluster_centers_

        if best_labels is None:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
            best_labels = kmeans.fit_predict(features)
            best_centers = kmeans.cluster_centers_

        keyframe_indices = []
        for i in range(len(best_centers)):
            cluster_frames = np.where(best_labels == i)[0]
            if len(cluster_frames) > 0:
                dists = np.linalg.norm(features[cluster_frames] - best_centers[i], axis=1)
                keyframe_indices.append(cluster_frames[np.argmin(dists)])

        keyframe_features = features[keyframe_indices]
        n_keyframes = len(keyframe_indices)
        if n_keyframes <= 1: return keyframe_indices

        similarity_matrix = np.dot(keyframe_features, keyframe_features.T)
        to_keep = list(range(n_keyframes))
        threshold = 0.92
        i = 0
        while i < len(to_keep):
            j = i + 1
            while j < len(to_keep):
                if similarity_matrix[to_keep[i], to_keep[j]] > threshold:
                    to_keep.pop(j)
                else: j += 1
            i += 1
        return [keyframe_indices[i] for i in to_keep]

    def _post_process_batch_concurrently(self, batch_data, tag_model, ocr_model):
        """
        Xử lý đồng thời một lô (batch) các keyframe.
        - OCR được thực hiện trong một lệnh gọi API duy nhất cho cả lô.
        - Tagging được thực hiện tuần tự cho từng ảnh trong lô.
        """
        if not batch_data:
            return []

        keyframe_paths, images = zip(*batch_data)
        batch_size = len(keyframe_paths)
        
        tags_results = [[] for _ in range(batch_size)]
        ocr_results = ["" for _ in range(batch_size)]
        
        # 1. <<< SỬA LỖI >>>: Thực hiện OCR theo lô với định dạng input đúng và phân tích JSON output
        if ocr_model and images:
            try:
                print(f"[PostProc] Submitting batch of {batch_size} frames for OCR...")
                # SỬA LỖI 1: Chuyển tuple 'images' thành list
                response_str, _ = ocr_model.generate(
                    model_name="gemini-2.0-flash", # Có thể thay đổi model nếu cần
                    images=list(images), # Chuyển đổi tuple thành list
                    prompt=OCR_PROMPT
                )
                
                # SỬA LỖI 2: Phân tích chuỗi JSON trả về
                # Làm sạch chuỗi JSON khỏi các ký tự không mong muốn như ```json ... ```
                clean_response_str = response_str.strip().replace('```json', '').replace('```', '').strip()
                response_json = json.loads(clean_response_str)
                
                extracted_texts = response_json.get("extracted_texts", [])

                if isinstance(extracted_texts, list) and len(extracted_texts) == batch_size:
                    ocr_results = [text.strip() for text in extracted_texts]
                else:
                    ocr_results = [text.strip() for text in extracted_texts]
                    ocr_rerulst += [""]*(batch_size-len(extracted_texts))
                    print(f"[WARNING] OCR batch result has unexpected format or length. Received texts: {extracted_texts}")

            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to decode JSON from OCR response: {e}\nResponse was: {response_str}")
            except Exception as e:
                print(f"[WARNING] OCR batch processing failed: {e}")

        # 2. Thực hiện Tagging cho từng ảnh trong lô
        if tag_model:
            for i in range(batch_size):
                try:
                    tags = RAM.get_tag(keyframe_paths[i], tag_model)
                    tags_results[i] = tags
                except Exception as e:
                    print(f"[WARNING] Tagging failed for {os.path.basename(keyframe_paths[i])}: {e}")

        # 3. Kết hợp kết quả và trả về
        final_results = []
        for i in range(batch_size):
            final_results.append((keyframe_paths[i], tags_results[i], ocr_results[i]))
            
        return final_results

    def _process_shot_concurrently_jit(self, shot_task_data: Tuple) -> Tuple:
        shot_idx, (start_frame_idx, end_frame_idx), video_path, sample_rate, max_frames_per_shot = shot_task_data

        # Lấy danh sách chỉ số frame cần xử lý cho shot này
        shot_original_frame_indices = list(range(start_frame_idx, end_frame_idx + 1, sample_rate))
        
        if len(shot_original_frame_indices) > max_frames_per_shot:
            step = len(shot_original_frame_indices) // max_frames_per_shot
            shot_original_frame_indices = shot_original_frame_indices[::step]

        if not shot_original_frame_indices:
            return shot_idx, [], []

        # Mở video BÊN TRONG luồng này
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR][Shot {shot_idx+1}] Could not open video file: {video_path}")
            return shot_idx, [], []

        shot_frames = []
        for frame_idx in shot_original_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                shot_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release() # Rất quan trọng: giải phóng tài nguyên ngay lập tức

        if not shot_frames:
            return shot_idx, [], []

        # Phần còn lại của logic không đổi
        shot_features = self.extract_clip_features(shot_frames, shot_id=shot_idx + 1)
        keyframe_indices_in_shot = self.adaptive_clustering(shot_features)
        
        # Trả về chỉ số gốc của các keyframe đã chọn
        selected_original_indices = [shot_original_frame_indices[i] for i in keyframe_indices_in_shot]
        selected_frames = [shot_frames[i] for i in keyframe_indices_in_shot]

        # Thay vì trả về chỉ số trong shot, trả về frame thật và chỉ số gốc
        return shot_idx, selected_frames, selected_original_indices
    

    # <<< MAIN METHOD REFACTORED FOR STREAMING PIPELINE >>>
    def extract_keyframes(self, video_path: str) -> None:
        # ... (các phần code khác không đổi) ...

        # Reset frame counter   for each new video
        self._frame_counter = 1

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)

        if os.path.exists(video_output_dir): shutil.rmtree(video_output_dir)
        os.makedirs(video_output_dir, exist_ok=True)

        new_video_path = os.path.join(video_output_dir, os.path.basename(video_path))
        shutil.copy(video_path, new_video_path)
        video_path = new_video_path
        print(f"\n[PIPELINE] Starting processing for: {video_path}")

        # --- STAGE 1 (Sequential): Shot Detection and Frame Extraction ---
        fps = self.get_video_fps(video_path)
        # SỬA LỖI: Chỉ lấy prediction, bỏ qua mảng video trả về
        _, single_frame_predictions, _ = self.transnet.predict_video(video_path)
        scenes = self.transnet.predictions_to_scenes(single_frame_predictions)
        print(f"[PIPELINE] Detected {len(scenes)} shots.")

        # LOAD RAM MODEL HERE, BEFORE ANY THREADS MIGHT NEED IT FOR POST-PROCESSING
        # (Đây là một sửa lỗi logic trong code ban đầu của bạn,
        # vì tag_model cần được tải trước khi các task post-processing được submit)
        # <<< THAY ĐỔI >>>: Tải các model (giống như trước)
    # <<< CẢI THIỆN >>>: Sử dụng model đã được tải trong __init__ một cách nhất quán
        tag_model = self.ram
        ocr_model = None
        try:
            # tag_model đã được tải
            print("[PIPELINE] RAM tag model is already loaded.")
        except Exception as e:
            print(f"[ERROR] RAM tag model check failed: {e}. Post-processing will skip tagging.")
            tag_model = None # Đảm bảo model là None nếu có lỗi
        try:
            ocr_model = Generator(base_url="http://localhost:9600", api_key=API_KEY)
            print("[PIPELINE] OCR model initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Could not initialize OCR model: {e}. Post-processing will skip OCR.")
        
        all_keyframes_info = []
        metadata_stubs = {}
        metadata = {video_name: {}}
        num_cpu_workers = min(4, os.cpu_count() or 1) 
        print(f"[PIPELINE] Using {num_cpu_workers} workers for shot processing (CPU/GPU overlap).")

        with ThreadPoolExecutor(...) as shot_executor, ...:
            shot_futures = {}
            for shot_idx, scene in enumerate(scenes):
                # SỬA LỖI: Truyền video_path và cảnh, không truyền all_frames
                future = shot_executor.submit(
                    self._process_shot_concurrently_jit, # Gọi hàm JIT mới
                    (shot_idx, scene, video_path, self.sample_rate, self.max_frames_per_shot)
                )
                shot_futures[future] = shot_idx
            post_processing_futures = []
            keyframe_batch_buffer = []

            print(f"\n[PIPELINE] Submitted {len(shot_futures)} shots for sequential processing. Starting streaming pipeline...")

            for future in as_completed(shot_futures):
                shot_idx = shot_futures[future]
                try:
                    s_idx, keyframe_indices_in_shot, shot_frame_indices = future.result()
                    if not keyframe_indices_in_shot:
                        continue
                    
                    print(f"[PIPELINE] Shot {s_idx+1} completed. Found {len(keyframe_indices_in_shot)} keyframes. Buffering for post-processing...")

                    for i, keyframe_idx_in_shot in enumerate(keyframe_indices_in_shot):
                        frame_count = self._get_next_frame_count()
                        original_sampled_idx = shot_frame_indices[keyframe_idx_in_shot]
                        original_frame_idx = all_frame_indices[original_sampled_idx]
                        
                        frame_filename = f"frame_{frame_count:03d}.webp"
                        keyframe_path = os.path.join(video_output_dir, frame_filename)

                        image = Image.fromarray(all_frames[original_sampled_idx])
                        image.save(keyframe_path, 'WEBP', quality=90)
                        
                        keyframe_batch_buffer.append((keyframe_path, image))
                        
                        if len(keyframe_batch_buffer) >= OCR_BATCH_SIZE:
                            print(f"[PIPELINE] Submitting a batch of {len(keyframe_batch_buffer)} keyframes for post-processing.")
                            task_future = post_proc_executor.submit(
                                self._post_process_batch_concurrently,
                                keyframe_batch_buffer[:],
                                tag_model,
                                ocr_model
                            )
                            post_processing_futures.append(task_future)
                            keyframe_batch_buffer.clear()
                        
                        timestamp = f"{(original_frame_idx / fps) // 60:02.0f}:{(original_frame_idx / fps) % 60:06.3f}"
                        metadata_stubs[keyframe_path] = {
                            "frame_name": f"frame_{frame_count:03d}",
                            "id": original_frame_idx,
                            "time-stamp": timestamp,
                            "shot": s_idx + 1,
                        }
                        all_keyframes_info.append((keyframe_path, s_idx, i, original_frame_idx))
                except Exception as exc:
                    print(f'[PIPELINE] Shot {shot_idx + 1} generated an exception: {exc}')

            if keyframe_batch_buffer:
                print(f"[PIPELINE] Submitting the final batch of {len(keyframe_batch_buffer)} keyframes.")
                task_future = post_proc_executor.submit(
                    self._post_process_batch_concurrently,
                    keyframe_batch_buffer[:],
                    tag_model,
                    ocr_model
                )
                post_processing_futures.append(task_future)
                keyframe_batch_buffer.clear()

            # <<< SỬA LỖI >>>: XÓA VÒNG LẶP THỨ HAI. Chỉ giữ lại vòng lặp ĐÚNG này.
            print(f"\n[PIPELINE] All shots processed. Aggregating results from {len(post_processing_futures)} batches...")
            for future in as_completed(post_processing_futures):
                try:
                    batch_results = future.result()
                    if not batch_results: continue # Bỏ qua nếu lô không có kết quả

                    for keyframe_path, tags, ocr_text in batch_results:
                        if keyframe_path in metadata_stubs:
                            stub = metadata_stubs[keyframe_path]
                            frame_name = stub.pop("frame_name")
                            metadata[video_name][frame_name] = {
                                **stub, 
                                "tags": tags,
                                "ocr": ocr_text
                            }
                except Exception as exc:
                    print(f'[PIPELINE] A post-processing batch generated an exception: {exc}')
            
        # --- STAGE 4 (Finalization): Save all files ---
        # Sort metadata by frame number for consistent output
        sorted_frames = sorted(metadata[video_name].items(), key=lambda item: int(item[0].split('_')[1]))
        metadata[video_name] = dict(sorted_frames)

        metadata_path = os.path.join(video_output_dir, "metadata.json")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=3, ensure_ascii=False)
        print(f"[PIPELINE] Metadata saved to {metadata_path}")
        
        # Sort summary info for readability
        all_keyframes_info.sort(key=lambda x: int(os.path.basename(x[0]).split('_')[1].split('.')[0]))
        with open(os.path.join(video_output_dir, "keyframes_summary.txt"), "w") as f:
            f.write(f"Video: {os.path.basename(video_path)}\nTotal shots: {len(scenes)}\nTotal keyframes: {len(all_keyframes_info)}\n\n")
            for path, s_idx, k_idx, o_idx in all_keyframes_info:
                f.write(f"{os.path.basename(path)}: Shot {s_idx+1}, Keyframe {k_idx+1}, Original Frame {o_idx}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from videos using a streaming pipeline.")
    parser.add_argument("video_path", type=str, help="Path to the video file or directory of videos")
    parser.add_argument("--output", type=str, default="keyframes", help="Output directory for keyframes")
    parser.add_argument("--sample-rate", type=int, default=5, help="Sample every N frames to reduce computation")
    parser.add_argument("--max-frames", type=int, default=55, help="Maximum number of frames to process per shot")
    args = parser.parse_args()

    start = time.time()
    extractor = VideoKeyframeExtractor(
        transnet_weights="transnetv2-weights",
        output_dir=args.output,
        sample_rate=args.sample_rate,
        max_frames_per_shot=args.max_frames
    )
    
    extractor.extract_keyframes(args.video_path)
    end = time.time()
    last = end - start
    print(f"Total time taken for keyframe extraction: {last:.2f} seconds")

if __name__ == "__main__":
    main()


    # python infer_concurent.py /workspace/WorkingSpace/Personal/chinhnm/data/L01_V002.mp4 --output /workspace/WorkingSpace/Personal/chinhnm/dmdm_demo