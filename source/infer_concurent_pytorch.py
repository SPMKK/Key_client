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
import source.RAM as RAM
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
from source.gemini_mistral_server import GeminiApiKeyManager, MistralApiKeyManager
from source.Generator import Generator
import torch
import torch.nn as nn
import torch.nn.functional as functional
import random

# (All class definitions like TransNetV2 and other helper methods remain the same)
# ... [TransNetV2 class and other methods from previous answer go here] ...
# NOTE: To keep the answer focused, I am omitting the unchanged code.
# Assume the TransNetV2 class and helper methods from the previous answer are present.
OCR_BATCH_SIZE = 6
MISTRAL_SHEET_URL = os.getenv("MISTRAL_SHEET_URL", "https://docs.google.com/spreadsheets/d/1NAlj7OiD9apH3U47RLJK0en1wLSW78X5zqmf6NmVUA4/export?format=csv&gid=0")
GEMINI_SHEET_URL = os.getenv("GEMINI_SHEET_URL", "https://docs.google.com/spreadsheets/d/1gqlLToS3OXPA-CvfgXRnZ1A6n32eXMTkXz4ghqZxe2I/export?format=csv&gid=0")

API_KEY = GeminiApiKeyManager(
    sheet_url=GEMINI_SHEET_URL
).get_active_key_count()
#prompt
API_KEY_MISTRAL = MistralApiKeyManager(
    sheet_url=MISTRAL_SHEET_URL
).get_active_key_count()
OCR_PROMPT = '''
You are an OCR agent specialized in extracting text in both **Vietnamese** and **English**.

Your task is to:

- Accurately detect and extract **all visible text** from each input image, including:
  - 📰 **Main headline / title**
  - 📝 **Subheadings / captions**
  - 📄 **Body text or any other readable content**
- Extract all text even if it is a logo, banner, or any other form of text.
- Preserve the **natural reading order** (top to bottom, left to right).
- Recognize both **Vietnamese (without diacritics)** and **English** text.
- **Do not translate** the text—just extract it **verbatim** as it appears.
- If an image contains **no readable text**, return an empty string for that frame.
- Input is a **batch of images**, typically extracted from a video (e.g., keyframes).
- Output the extracted texts without diacritics in a **structured format**, preserving the input order.
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

class TransNetV2(nn.Module):

    def __init__(self,
                 F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5,
                 use_convex_comb_reg=False,  # not supported
                 use_resnet_features=False,  # not supported
                 use_resnet_like_top=False,  # not supported
                 frame_similarity_on_last_layer=False):  # not supported
        super(TransNetV2, self).__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplemented("Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True
        ) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(
            lookup_window=101, output_dim=128
        ) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity: output_dim += 128
        if use_color_histograms: output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, \
            "incorrect input type and/or shape"
        # uint8 of shape [B, T, H, W, 3] to float of shape [B, 3, T, H, W]
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot

class StackedDDCNNV2(nn.Module):

    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,
                 use_octave_conv=False,  # not supported
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()

        if use_octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, octave_conv=use_octave_conv,
                          activation=functional.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)
        ])
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        x = self.pool(x)
        return x


class DilatedDCNNV2(nn.Module):

    def __init__(self,
                 in_filters,
                 filters,
                 batch_norm=True,
                 activation=None,
                 octave_conv=False):  # not supported
        super(DilatedDCNNV2, self).__init__()

        if octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class Conv3DConfigurable(nn.Module):

    def __init__(self,
                 in_filters,
                 filters,
                 dilation_rate,
                 separable=True,
                 octave=False,  # not supported
                 use_bias=True,
                 kernel_initializer=None):  # not supported
        super(Conv3DConfigurable, self).__init__()

        if octave:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")
        if kernel_initializer is not None:
            raise NotImplemented("Kernel initializers are not implemented in Pytorch version of Transnet!")

        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                              dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            conv = nn.Conv3d(in_filters, filters, kernel_size=3,
                             dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class FrameSimilarity(nn.Module):

    def __init__(self,
                 in_filters,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 stop_gradient=False,  # not supported
                 use_bias=False):
        super(FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplemented("Stop gradient not implemented in Pytorch version of Transnet!")

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        x = torch.transpose(x, 1, 2)

        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        return functional.relu(self.fc(similarities))


class ColorHistograms(nn.Module):

    def __init__(self,
                 lookup_window=101,
                 output_dim=None):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = frames.int()

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities

class VideoKeyframeExtractor:
    def __init__(self, transnet_weights=None, output_dir="keyframes", 
                 sample_rate=1, max_frames_per_shot=16, model="google/siglip2-base-patch16-512", base_url = "localhost:6660"):
        self.device = "cuda"

        print("[TransNetV2-PyTorch] Initializing model.")
        self.transnet = TransNetV2()
         # Kiểm tra và tải trọng số
        if transnet_weights is None or not os.path.exists(transnet_weights):
            raise FileNotFoundError(f"PyTorch weights for TransNetV2 not found at: {transnet_weights}")
        
        print(f"[TransNetV2-PyTorch] Loading weights from {transnet_weights}")
        state_dict = torch.load(transnet_weights, map_location=self.device)
        self.transnet.load_state_dict(state_dict)
        self.transnet.to(self.device) # Chuyển model lên GPU/CPU
        self.transnet.eval() # Đặt model ở chế độ đánh giá
        print("[TransNetV2-PyTorch] Model loaded successfully.")

        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.ram = RAM.load_tag_model()
        self.ocr_model = Generator(
            base_url= 'http://0.0.0.0:9600',
            api_key=API_KEY
        )
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.max_frames_per_shot = max_frames_per_shot
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
                batch_frames = frames[i:i+batch_size]

                # Convert frames to PIL Images and preprocess for CLIP
                batch_inputs = torch.stack([
                    self.preprocess(Image.fromarray(frame))
                    for frame in batch_frames
                ]).to(self.device)

                # Extract features
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_inputs)
                    batch_features /= batch_features.norm(dim=-1, keepdim=True)  # Normalize

                features.append(batch_features.cpu().numpy())
                print(f"\r[CLIP] Processing frames {i+len(batch_frames)}/{len(frames)}", end="")
                
                # Giải phóng bộ nhớ GPU sau mỗi batch nếu có thể
                del batch_features, batch_inputs
                torch.cuda.empty_cache() 
                # print(f"\r{progress_prefix} Processing frames {i + len(batch_frames)}/{len(frames)}", end="")
            
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
        threshold = 0.94
        i = 0
        while i < len(to_keep):
            j = i + 1
            while j < len(to_keep):
                if similarity_matrix[to_keep[i], to_keep[j]] > threshold:
                    to_keep.pop(j)
                else: j += 1
            i += 1
        return [keyframe_indices[i] for i in to_keep]
    
    def _extract_json_from_response(self, response_str: str) -> dict | None:
        """
        Safely extracts a JSON object from a string that may be wrapped in markdown or other text.
        Finds the first '{' and the last '}' to delimit the JSON object.
        """
        try:
            # Find the starting position of the JSON object
            start_index = response_str.find('{')
            # Find the ending position of the JSON object
            end_index = response_str.rfind('}')

            if start_index == -1 or end_index == -1 or end_index < start_index:
                # If no valid JSON object is found, return None
                print(f"[JSON_HELPER] Could not find a valid JSON object in the response.")
                return None

            # Extract the JSON substring
            json_str = response_str[start_index : end_index + 1]

            # Parse the extracted string
            return json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"[JSON_HELPER] Failed to decode the extracted JSON string: {e}")
            return None
        except Exception as e:
            print(f"[JSON_HELPER] An unexpected error occurred during JSON extraction: {e}")
        return None
    
    def _post_process_batch_concurrently(self, batch_data, tag_model, ocr_model, caption_model=None):
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
        captions = ["" for _ in range(batch_size)]
        # 1. <<< SỬA LỖI >>>: Thực hiện OCR theo lô với định dạng input đúng và phân tích JSON output
        if ocr_model and images:
            try:
                print(f"[PostProc] Submitting batch of {batch_size} frames for OCR...")
                # SỬA LỖI 1: Chuyển tuple 'images' thành list
                response_str, _ = ocr_model.generate(
                    model_name="gemini-2.5-flash", # Có thể thay đổi model nếu cần
                    images=list(images), # Chuyển đổi tuple thành list
                    prompt=OCR_PROMPT,
                    temperature=0.0,
                )
                
                # SỬA LỖI 2: Phân tích chuỗi JSON trả về
                # Làm sạch chuỗi JSON khỏi các ký tự không mong muốn như ```json ... ```
            # Use the robust helper to extract and parse JSON
                response_json = self._extract_json_from_response(response_str)

                if response_json and "extracted_texts" in response_json:
                    raw_texts = response_json.get("extracted_texts", [])
                    print(f"[PostProc] Successfully parsed {len(raw_texts)} texts from OCR response.")
                    
                    # Ensure the final ocr_results list matches the batch size
                    num_returned = len(raw_texts)
                    for i in range(batch_size):
                        if i < num_returned and raw_texts[i]:
                            ocr_results[i] = raw_texts[i].strip()
                        else:
                            ocr_results[i] = "" # Fill with empty string if not provided or empty
                    
                    if num_returned != batch_size:
                        print(f"[WARNING] OCR returned {num_returned} results for a batch of {batch_size}. Padding/truncating as needed.")

                else:
                    # Handles cases where parsing failed or the 'extracted_texts' key is missing
                    print(f"[WARNING] OCR response could not be parsed or was invalid. Using empty texts. Full response: {response_str}")
                    # 'ocr_results' is already initialized to empty strings, so no action needed

            except Exception as e:
                # Fallback for generate() call failures (e.g., network error)
                print(f"[WARNING] OCR batch processing failed with an API/network error: {e}. All OCR results for this batch will be empty.")

                
        if caption_model:
            try:
                print(f"[PostProc] Submitting batch of {batch_size} frames for captions...")
                # Convert tuple 'images' to a list for the API call
                with open('source/prompt_caption.txt', 'r') as f:
                    prompt_caption = f.read().strip()

                response_str, _ = caption_model.generate(
                    model_name="pixtral-large-latest", # Using a reliable model
                    images=list(images), 
                    prompt=prompt_caption,
                    temperature=0.2,
                )
                
                # Robustly parse the JSON response
                response_json = self._extract_json_from_response(response_str)

                if response_json and "captions" in response_json:
                    raw_captions = response_json.get("captions", [])
                    print(f"[PostProc] Successfully parsed {len(raw_captions)} captions from response.")
                    
                    # Ensure the final captions list matches the batch size
                    num_returned = len(raw_captions)
                    for i in range(batch_size):
                        if i < num_returned and raw_captions[i]:
                            captions[i] = raw_captions[i].strip()
                        else:
                            captions[i] = "" # Fill with empty string if not provided or empty

                    if num_returned != batch_size:
                        print(f"[WARNING] Captioning returned {num_returned} results for a batch of {batch_size}. Padding/truncating as needed.")
                
                else:
                    # This branch handles cases where parsing failed or the 'captions' key is missing
                    print(f"[WARNING] Captioning response could not be parsed or was invalid. Using empty captions. Full response: {response_str}")
                    # 'captions' is already initialized to empty strings, so no action needed

            except Exception as e:
                # This is a fallback for generate() call failures (e.g., network error)
                print(f"[WARNING] Captioning batch processing failed with an API/network error: {e}. All captions for this batch will be empty.")
        # 2. Thực hiện Tagging cho từng ảnh trong lô
        if tag_model:
            # with self._gpu_lock:
            for i in range(batch_size):
                try:
                    tags = RAM.get_tag(keyframe_paths[i], tag_model)
                    tags_results[i] = tags
                except Exception as e:
                    print(f"[WARNING] Tagging failed for {os.path.basename(keyframe_paths[i])}: {e}")

        # 3. Kết hợp kết quả và trả về
        final_results = []
        for i in range(batch_size):
            final_results.append((keyframe_paths[i], tags_results[i], ocr_results[i], captions[i]))
            
        return final_results

    def _process_shot_concurrently(self, shot_task_data: Tuple) -> Tuple:
        shot_idx, (start_idx, end_idx), all_frames, all_frame_indices = shot_task_data
        shot_frame_indices = [
            i for i, frame_idx in enumerate(all_frame_indices)
            if start_idx <= frame_idx <= end_idx
        ]
        if len(shot_frame_indices) > self.max_frames_per_shot:
            step = len(shot_frame_indices) // self.max_frames_per_shot
            shot_frame_indices = shot_frame_indices[::step]
        if not shot_frame_indices:
            return shot_idx, [], []
        shot_frames = [all_frames[i] for i in shot_frame_indices]
        shot_features = self.extract_clip_features(shot_frames, shot_id=shot_idx + 1)
        keyframe_indices_in_shot = self.adaptive_clustering(shot_features)
        return shot_idx, keyframe_indices_in_shot, shot_frame_indices
    
    @staticmethod
    def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.3): # Tôi đã đổi threshold mặc định thành 0.3 cho giống pipeline của bạn
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

    def _predict_frames_pytorch(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Xử lý các frame đã được trích xuất bằng model TransNetV2 PyTorch.
            Hàm này mô phỏng lại logic cửa sổ trượt của phiên bản gốc.
            """
            # Chuyển model sang chế độ đánh giá
            self.transnet.eval()
            
            predictions = []
            
            # Logic cửa sổ trượt (sliding window) - được giữ nguyên từ phiên bản TensorFlow
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

            for batch_np in input_iterator():
                # Chuyển đổi NumPy sang Tensor, đưa lên GPU
                batch_torch = torch.from_numpy(batch_np).to(self.device)
                
                with torch.no_grad(): # Quan trọng: Vô hiệu hóa tính toán gradient để tiết kiệm bộ nhớ và tăng tốc
                    # Forward pass
                    single_frame_pred_logits, all_frames_pred_logits_dict = self.transnet(batch_torch)
                    
                    # Áp dụng sigmoid để có được xác suất
                    single_frame_pred = torch.sigmoid(single_frame_pred_logits)
                    all_frames_pred = torch.sigmoid(all_frames_pred_logits_dict["many_hot"])

                # Chuyển kết quả về CPU, sang NumPy và cắt lấy phần dự đoán đáng tin cậy (25:75)
                # .detach() để tách khỏi computational graph
                predictions.append((
                    single_frame_pred.detach().cpu().numpy()[0, 25:75, 0],
                    all_frames_pred.detach().cpu().numpy()[0, 25:75, 0]
                ))

                print(f"\r[TransNetV2-PyTorch] Processing video frames {min(len(predictions) * 50, len(frames))}/{len(frames)}", end="")
            print("")

            single_frame_pred_np = np.concatenate([p[0] for p in predictions])
            all_frames_pred_np = np.concatenate([p[1] for p in predictions])

            return single_frame_pred_np[:len(frames)], all_frames_pred_np[:len(frames)]

    def _predict_video_pytorch(self, video_fn: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Trích xuất frame từ video và chạy dự đoán bằng model PyTorch.
            """
            try:
                import ffmpeg
            except ModuleNotFoundError:
                raise ModuleNotFoundError("`ffmpeg-python` is required. Please run `pip install ffmpeg-python`")

            print(f"[TransNetV2-PyTorch] Extracting frames from {video_fn}")
            video_stream, _ = ffmpeg.input(video_fn).output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
            ).run(capture_stdout=True, capture_stderr=True)

            video_frames = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
            
            single_pred, all_pred = self._predict_frames_pytorch(video_frames)
            return video_frames, single_pred, all_pred
    def _validate_and_rerun_missing_frames(
        self,
        video_output_dir: str,
        metadata: dict,
        all_keyframes_info: List[Tuple],
        fps: float,
        tag_model,
        ocr_model,
        caption_model
    ) -> dict:
        """
        Validates that all saved keyframe images have corresponding metadata entries.
        If any are missing, it re-runs the post-processing for them and updates the metadata.
        """
        print("\n[VALIDATION] Starting final check for missing metadata...")
        video_name = list(metadata.keys())[0]

        # 1. Identify all saved image files
        try:
            saved_files = {
                os.path.splitext(f)[0] for f in os.listdir(video_output_dir) if f.endswith('.webp')
            }
        except FileNotFoundError:
            print("[VALIDATION] Output directory not found. Skipping validation.")
            return metadata

        # 2. Identify all frames with existing metadata
        frames_in_metadata = set(metadata[video_name].keys())

        # 3. Find the difference
        missing_frame_names = sorted(list(saved_files - frames_in_metadata))

        if not missing_frame_names:
            print("[VALIDATION] All saved frames have metadata. No rerun needed.")
            return metadata

        print(f"[VALIDATION] Found {len(missing_frame_names)} frames with missing metadata: {missing_frame_names}")
        print("[VALIDATION] Preparing to rerun post-processing for these frames...")

        # 4. Prepare a batch for the rerun
        rerun_batch_data = []
        for frame_name in missing_frame_names:
            keyframe_path = os.path.join(video_output_dir, f"{frame_name}.webp")
            try:
                image = Image.open(keyframe_path)
                rerun_batch_data.append((keyframe_path, image))
            except FileNotFoundError:
                print(f"[VALIDATION] WARNING: Could not find image file for missing frame: {keyframe_path}")
            except Exception as e:
                print(f"[VALIDATION] WARNING: Could not load image {keyframe_path}: {e}")

        if not rerun_batch_data:
            print("[VALIDATION] No valid images to rerun. Aborting validation.")
            return metadata

        # 5. Execute the rerun (synchronously, as this is a final cleanup)
        print(f"[VALIDATION] Submitting a batch of {len(rerun_batch_data)} frames for rerun...")
        try:
            rerun_results = self._post_process_batch_concurrently(
                rerun_batch_data, tag_model, ocr_model, caption_model
            )
        except Exception as e:
            print(f"[VALIDATION] Rerun failed with a critical error: {e}")
            return metadata # Return original metadata to avoid data loss

        # 6. Integrate the results back into the main metadata object
        # Create a lookup for original frame info (shot, timestamp, etc.)
        info_lookup = {
            os.path.basename(path): (s_idx, o_idx)
            for path, s_idx, _, o_idx in all_keyframes_info
        }

        successful_reruns = 0
        for keyframe_path, tags, ocr_text, caption in rerun_results:
            frame_basename = os.path.basename(keyframe_path)
            frame_name_no_ext = os.path.splitext(frame_basename)[0]

            # Reconstruct the original stub information
            if frame_basename in info_lookup:
                s_idx, original_frame_idx = info_lookup[frame_basename]
                timestamp = f"{(original_frame_idx / fps) // 60:02.0f}:{(original_frame_idx / fps) % 60:06.3f}"
                
                metadata[video_name][frame_name_no_ext] = {
                    "id": original_frame_idx,
                    "time-stamp": timestamp,
                    "shot": s_idx + 1,
                    "tags": tags,
                    "ocr": ocr_text,
                    "caption": caption
                }
                successful_reruns += 1
            else:
                # Fallback if original info was somehow lost
                print(f"[VALIDATION] WARNING: Could not find original info for {frame_basename}. Creating partial metadata.")
                metadata[video_name][frame_name_no_ext] = {
                    "id": "unknown",
                    "time-stamp": "unknown",
                    "shot": "unknown",
                    "tags": tags,
                    "ocr": ocr_text,
                    "caption": caption
                }
    # <<< MAIN METHOD REFACTORED FOR STREAMING PIPELINE >>>
    def extract_keyframes(self, video_path: str) -> None:
        # ... (các phần code khác không đổi) ...

        # Reset frame counter   for each new video
        self._frame_counter = 1

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_dir, video_name)

        if os.path.exists(video_output_dir): shutil.rmtree(video_output_dir)
        os.makedirs(video_output_dir, exist_ok=True)

        # new_video_path = os.path.join(video_output_dir, os.path.basename(video_path))
        # shutil.copy(video_path, new_video_path)
        # video_path = new_video_path
        print(f"\n[PIPELINE] Starting processing for: {video_path}")

        # --- STAGE 1 (Sequential): Shot Detection and Frame Extraction ---
        fps = self.get_video_fps(video_path)
        _, single_frame_predictions, all_frame_predictions = self._predict_video_pytorch(video_path)
        scenes = self.predictions_to_scenes(single_frame_predictions, threshold=0.3)

        print(f"[PIPELINE] Detected {len(scenes)} shots.")
        print("[PIPELINE] Extracting all frames...")
        all_frames, all_frame_indices = self.extract_video_frames(video_path)
        print(f"[PIPELINE] Extracted {len(all_frames)} frames.")

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
            ocr_model = self.ocr_model
            print("[PIPELINE] OCR model initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Could not initialize OCR model: {e}. Post-processing will skip OCR.")
        try:
            caption_model = Generator(base_url='http://0.0.0.0:9600', api_key=API_KEY_MISTRAL)
            print("[PIPELINE] Caption model initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Could not initialize Caption model: {e}. Post-processing will skip captioning.")
        
        all_keyframes_info = []
        metadata_stubs = {}
        metadata = {video_name: {}}
        num_cpu_workers = min(4, os.cpu_count() or 1) 
        print(f"[PIPELINE] Using {num_cpu_workers} workers for shot processing (CPU/GPU overlap).")

        with ThreadPoolExecutor(max_workers=num_cpu_workers, thread_name_prefix='ShotProc') as shot_executor, \
             ThreadPoolExecutor(max_workers=12, thread_name_prefix='PostProc') as post_proc_executor:

            shot_futures = {
                shot_executor.submit(
                    self._process_shot_concurrently,
                    (shot_idx, scene, all_frames, all_frame_indices)
                ): shot_idx
                for shot_idx, scene in enumerate(scenes)
            }
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

                    # This inner loop is where the fix is applied
                    for i, keyframe_idx_in_shot in enumerate(keyframe_indices_in_shot):
                        frame_count = self._get_next_frame_count()
                        original_sampled_idx = shot_frame_indices[keyframe_idx_in_shot]
                        original_frame_idx = all_frame_indices[original_sampled_idx]
                        
                        frame_filename = f"frame_{frame_count:03d}.webp"
                        keyframe_path = os.path.join(video_output_dir, frame_filename)

                        image = Image.fromarray(all_frames[original_sampled_idx])
                        
                        # =============================================================
                        # <<< FIX: CREATE METADATA STUB *BEFORE* BUFFERING THE FRAME >>>
                        # By creating the stub first, we guarantee its existence before
                        # the frame can ever be part of a submitted batch.
                        # =============================================================
                        timestamp = f"{(original_frame_idx / fps) // 60:02.0f}:{(original_frame_idx / fps) % 60:06.3f}"
                        metadata_stubs[keyframe_path] = {
                            "frame_name": f"frame_{frame_count:03d}",
                            "id": original_frame_idx,
                            "time-stamp": timestamp,
                            "shot": s_idx + 1,
                        }

                        # Now it's safe to save the image and add to the buffer
                        image.save(keyframe_path, 'WEBP', quality=90)
                        keyframe_batch_buffer.append((keyframe_path, image))
                        
                        # The batch submission logic is now safe
                        if len(keyframe_batch_buffer) >= OCR_BATCH_SIZE:
                            print(f"[PIPELINE] Submitting a batch of {len(keyframe_batch_buffer)} keyframes for post-processing.")
                            
                            # Best practice: create a new list for the task and re-initialize the buffer
                            # This is slightly safer than using a slice `[:]` and `clear()`.
                            task_batch = keyframe_batch_buffer
                            keyframe_batch_buffer = []

                            task_future = post_proc_executor.submit(
                                self._post_process_batch_concurrently,
                                task_batch, # Submit the captured batch
                                tag_model,
                                ocr_model,
                                caption_model
                            )
                            post_processing_futures.append(task_future)
                        
                        # The all_keyframes_info list is for the summary.txt, its position is less critical.
                        all_keyframes_info.append((keyframe_path, s_idx, i, original_frame_idx))

                except Exception as exc:
                    print(f'[PIPELINE] Shot {shot_idx + 1} generated an exception: {exc}')

            if keyframe_batch_buffer:
                print(f"[PIPELINE] Submitting the final batch of {len(keyframe_batch_buffer)} keyframes.")
                task_future = post_proc_executor.submit(
                    self._post_process_batch_concurrently,
                    keyframe_batch_buffer, # Submit the final buffer
                    tag_model,
                    ocr_model,
                    caption_model
                )
                post_processing_futures.append(task_future)

            # <<< SỬA LỖI >>>: XÓA VÒNG LẶP THỨ HAI. Chỉ giữ lại vòng lặp ĐÚNG này.
            print(f"\n[PIPELINE] All shots processed. Aggregating results from {len(post_processing_futures)} batches...")
            for future in as_completed(post_processing_futures):
                try:
                    batch_results = future.result()
                    if not batch_results: continue

                    for keyframe_path, tags, ocr_text, caption in batch_results:
                        # This check will no longer fail due to the race condition
                        if keyframe_path in metadata_stubs:
                            stub = metadata_stubs[keyframe_path]
                            frame_name = stub.pop("frame_name")
                            metadata[video_name][frame_name] = {
                                **stub, 
                                "tags": tags,
                                "ocr": ocr_text,
                                "caption": caption
                            }
                        else:
                            # This warning can help debug other potential issues
                            print(f"[WARNING] Could not find metadata stub for processed keyframe: {keyframe_path}")
                except Exception as exc:
                    print(f'[PIPELINE] A post-processing batch generated an exception: {exc}')
        metadata = self._validate_and_rerun_missing_frames(
            video_output_dir,
            metadata,
            all_keyframes_info,
            fps,
            tag_model,
            ocr_model,
            caption_model
        )
        
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
        transnet_weights="transnetv2-pytorch-weights.pth",
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