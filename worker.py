# worker.py
import argparse
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Literal, List
import base64
import zipfile
import requests

# ===== 0) Import extractor =====
try:
    from source.infer_concurent_pytorch import VideoKeyframeExtractor
except ImportError as e:
    logging.error("Không thể import VideoKeyframeExtractor từ infer_concurrent_autoshot.py.")
    raise e


# ===== 1) Config =====
class WorkerConfig:
    WORKER_ID: str = f"worker-{os.getpid()}"
    WORKING_DIR: Path = Path("./worker_temp")

    SAMPLE_RATE: int = 5
    MAX_FRAMES_PER_SHOT: int = 55

    POLL_INTERVAL_SECONDS: int = 10

    # Số ảnh gửi mỗi batch base64
    BASE64_BATCH_SIZE: int = 8

    # Timeout HTTP
    REQ_TIMEOUT_GET: int = 15
    REQ_TIMEOUT_UPLOAD: int = 180
    REQ_TIMEOUT_REPORT: int = 30


# ===== 2) Logging =====
def setup_logging(worker_id: str):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {worker_id} - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"{worker_id}.log"),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ===== 3) Server I/O =====
def get_task_from_server(server_url: str, mode: str) -> Optional[Dict]:
    url = f"{server_url.rstrip('/')}/get_batch"
    payload = {"size": 1, "mode": mode}
    try:
        logging.info(f"Hỏi task mới (mode={mode}) -> {url}")
        r = requests.post(url, json=payload, timeout=WorkerConfig.REQ_TIMEOUT_GET)
        r.raise_for_status()
        tasks = r.json()
        if not tasks:
            logging.info("Chưa có task. Chờ...")
            return None
        task = tasks[0]
        logging.info(f"Nhận task: {task}")
        return task
    except requests.RequestException as e:
        logging.error(f"Lỗi kết nối get_batch: {e}")
        return None
    except Exception as e:
        logging.error(f"Lỗi không xác định get_batch: {e}")
        return None


def report_completion_to_server(server_url: str, video_id: str) -> bool:
    url = f"{server_url.rstrip('/')}/report_done"
    try:
        logging.info(f"Báo COMPLETED -> {url} (video_id={video_id})")
        r = requests.post(url, params={"video_id": video_id}, timeout=WorkerConfig.REQ_TIMEOUT_REPORT)
        r.raise_for_status()
        logging.info(f"Server xác nhận: {r.json()}")
        return True
    except requests.RequestException as e:
        logging.error(f"Lỗi báo hoàn thành {video_id}: {e}")
        if getattr(e, "response", None) is not None:
            logging.error(f"Chi tiết server: {e.response.text}")
        return False


# ===== 4) Chuẩn hoá & đóng gói kết quả =====
def _list_webps(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".webp"])


def normalize_result_layout(root_output_dir: Path, video_id: str) -> Optional[Path]:
    """
    Đưa kết quả về layout mà server đang validate:
    <root>/<video_id>/
      - metadata.json
      - *.webp   (KHÔNG dùng keyframes/)
    Trả về đường dẫn thư mục <root>/<video_id> nếu hợp lệ, ngược lại None.
    """
    vid_dir = root_output_dir / video_id
    if not vid_dir.exists():
        logging.error(f"Thiếu thư mục kết quả cho {video_id}: {vid_dir}")
        return None

    # Nếu extractor tạo subdir 'keyframes', chuyển *.webp lên thẳng <video_id>/
    keyframes_dir = vid_dir / "keyframes"
    if keyframes_dir.exists() and keyframes_dir.is_dir():
        for wp in _list_webps(keyframes_dir):
            dest = vid_dir / wp.name
            try:
                if dest.exists():
                    dest.unlink()
                shutil.move(str(wp), str(dest))
            except Exception as e:
                logging.error(f"Lỗi move {wp.name} -> {dest}: {e}")
                return None
        # dọn thư mục rỗng nếu cần
        try:
            if not any(keyframes_dir.iterdir()):
                keyframes_dir.rmdir()
        except Exception:
            pass

    metadata = vid_dir / "metadata.json"
    if not metadata.exists():
        logging.error(f"Thiếu metadata.json trong {vid_dir}")
        return None

    # Chỉ giữ file .webp ở ngay dưới <video_id>/
    webps = _list_webps(vid_dir)
    if not webps:
        logging.error(f"Không tìm thấy file .webp nào trong {vid_dir}")
        return None

    logging.info(f"Chuẩn hoá layout OK: {vid_dir} (webp={len(webps)})")
    return vid_dir


def create_flat_zip_for_server(source_dir: Path, video_id: str, dest_dir: Path) -> Optional[Path]:
    """
    Đóng ZIP ở dạng FLAT: metadata.json + *.webp ngay ở root của ZIP (khớp /upload_result).
    """
    vid_dir = normalize_result_layout(source_dir, video_id)
    if vid_dir is None:
        return None

    zip_path = dest_dir / f"{video_id}_result.zip"
    try:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # metadata.json
            zf.write(vid_dir / "metadata.json", arcname="metadata.json")
            # *.webp
            for wp in _list_webps(vid_dir):
                zf.write(wp, arcname=wp.name)
        logging.info(f"Tạo ZIP thành công: {zip_path}")
        return zip_path
    except Exception as e:
        logging.error(f"Lỗi tạo ZIP: {e}")
        return None


# ===== 5) Upload =====
def upload_results_base64_batched(server_url: str, video_id: str, root_output_dir: Path) -> bool:
    """
    Gửi theo batch Base64 tới /upload_result_base64
    """
    url = f"{server_url.rstrip('/')}/upload_result_base64"
    vid_dir = normalize_result_layout(root_output_dir, video_id)
    if vid_dir is None:
        return False

    metadata_path = vid_dir / "metadata.json"
    webps = _list_webps(vid_dir)

    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except Exception as e:
        logging.error(f"Đọc metadata.json lỗi: {e}")
        return False

    total = len(webps)
    batch = WorkerConfig.BASE64_BATCH_SIZE
    num_batches = max(1, (total + batch - 1) // batch)
    logging.info(f"Gửi {total} webp trong {num_batches} batch (video_id={video_id})")

    for i in range(num_batches):
        start, end = i * batch, min(total, (i + 1) * batch)
        encoded_keyframes: Dict[str, str] = {}
        for img_path in webps[start:end]:
            try:
                encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
                encoded_keyframes[img_path.name] = encoded
            except Exception as e:
                logging.error(f"Base64 lỗi {img_path.name}: {e}")

        payload = {
            "video_id": video_id,
            "metadata_json": metadata_text if i == 0 else None,
            "keyframes": encoded_keyframes,
            "is_final_batch": (i == num_batches - 1),
        }

        try:
            r = requests.post(url, json=payload, timeout=WorkerConfig.REQ_TIMEOUT_UPLOAD)
            r.raise_for_status()
            logging.info(f"Batch {i+1}/{num_batches}: {r.json().get('message','OK')}")
        except requests.RequestException as e:
            logging.error(f"Upload batch {i+1} lỗi: {e}")
            if getattr(e, "response", None) is not None:
                logging.error(f"Chi tiết server: {e.response.text}")
            return False

    logging.info(f"Upload Base64 hoàn tất cho {video_id}.")
    return True


def upload_zip_to_server(server_url: str, video_id: str, zip_path: Path) -> bool:
    url = f"{server_url.rstrip('/')}/upload_result"
    try:
        with open(zip_path, "rb") as f:
            files = {"file": (zip_path.name, f, "application/zip")}
            data = {"video_id": video_id}
            r = requests.post(url, files=files, data=data, timeout=WorkerConfig.REQ_TIMEOUT_UPLOAD)
            r.raise_for_status()
        logging.info(f"Upload ZIP OK: {video_id}")
        return True
    except requests.RequestException as e:
        logging.error(f"Upload ZIP lỗi {video_id}: {e}")
        if getattr(e, "response", None) is not None:
            logging.error(f"Chi tiết server: {e.response.text}")
        return False


# ===== 6) Main loop =====
def main_loop(server_url: str, videos_dir: Path, mode: Literal['local', 'colab'], base_url: str):
    logging.info(f"Worker chạy ở chế độ: {mode.upper()}")
    WorkerConfig.WORKING_DIR.mkdir(exist_ok=True)

    # Khởi tạo extractor một lần
    logging.info("Khởi tạo VideoKeyframeExtractor...")
    extractor = VideoKeyframeExtractor(
        transnet_weights="transnetv2-pytorch-weights.pth",
        output_dir=str(WorkerConfig.WORKING_DIR),
        sample_rate=WorkerConfig.SAMPLE_RATE,
        max_frames_per_shot=WorkerConfig.MAX_FRAMES_PER_SHOT,
        base_url=base_url
    )
    logging.info("Extractor sẵn sàng.")

    while True:
        task = get_task_from_server(server_url, mode)
        if not task:
            time.sleep(WorkerConfig.POLL_INTERVAL_SECONDS)
            continue

        video_id = task["video_id"]
        filename = task["filename"]
        src_video = videos_dir / filename

        if not src_video.exists():
            logging.error(f"Không thấy video nguồn: {src_video}. Bỏ qua.")
            time.sleep(2)
            continue

        start = time.time()
        zip_path: Optional[Path] = None
        temp_task_dir: Optional[Path] = None

        try:
            logging.info(f"=== Bắt đầu xử lý: {video_id} ===")

            if mode == "local":
                # Server trả sẵn result_path (đường dẫn tuyệt đối muốn ghi)
                result_path = Path(task.get("result_path"))
                if not result_path:
                    logging.error("Thiếu result_path từ server (mode=local).")
                    continue
                result_path.mkdir(parents=True, exist_ok=True)

                # Ghi thẳng vào result_path
                extractor.output_dir = str(result_path.parent)  # extractor có thể tạo subdir <video_id>, ta chỉnh lại:
                extractor.output_dir = str(result_path)         # => đảm bảo file nằm trong chính result_path
                extractor.extract_keyframes(str(src_video))

                # Chuẩn hoá layout ngay tại result_path
                if normalize_result_layout(result_path.parent if result_path.name == video_id else result_path, video_id) is None:
                    logging.error("Chuẩn hoá layout thất bại (local).")
                    continue

                # Báo server validate + COMPLETE
                report_completion_to_server(server_url, video_id)

            else:
                # COLAB: ghi ra thư mục tạm, sau đó upload Base64
                extractor.output_dir = str(WorkerConfig.WORKING_DIR)
                extractor.extract_keyframes(str(src_video))

                temp_task_dir = WorkerConfig.WORKING_DIR / video_id
                if normalize_result_layout(WorkerConfig.WORKING_DIR, video_id) is None:
                    logging.error("Chuẩn hoá layout thất bại (colab).")
                    continue

                # Ưu tiên Base64 theo thiết kế server
                ok = upload_results_base64_batched(server_url, video_id, WorkerConfig.WORKING_DIR)
                if not ok:
                    logging.error("Upload Base64 thất bại. Thử phương án ZIP (fallback).")
                    # Fallback: ZIP
                    zip_path = create_flat_zip_for_server(WorkerConfig.WORKING_DIR, video_id, WorkerConfig.WORKING_DIR)
                    if zip_path:
                        upload_zip_to_server(server_url, video_id, zip_path)

            logging.info(f"Xử lý '{video_id}' xong trong {time.time() - start:.2f}s")

        except Exception as e:
            logging.error(f"Lỗi xử lý task {video_id}: {e}", exc_info=True)

        finally:
            # Dọn dẹp
            if zip_path and zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:
                    pass
            if mode == "colab" and temp_task_dir and temp_task_dir.exists():
                try:
                    shutil.rmtree(temp_task_dir, ignore_errors=True)
                except Exception:
                    pass
            time.sleep(2)


# ===== 7) Entry =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Worker xử lý video, trích xuất keyframe và gửi kết quả về server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:6661",
                        help="URL Task Dispatcher Server (vd: http://host:6661)")
    parser.add_argument("--videos-dir", type=str, required=True,
                        help="Thư mục chứa video nguồn (share được cho worker).")
    parser.add_argument("--mode", type=str, choices=["local", "colab"], default="colab",
                        help="local: ghi trực tiếp vào result_path và /report_done; colab: upload Base64.")
    parser.add_argument("--base-url", type=str, default="http://159.223.81.229:8000",
                        help="Tham số truyền cho extractor (nếu có dùng).")

    args = parser.parse_args()
    videos_dir_path = Path(args.videos_dir)
    if not videos_dir_path.exists() or not videos_dir_path.is_dir():
        print(f"Lỗi: Thư mục videos '{videos_dir_path}' không tồn tại hoặc không phải thư mục.")
        exit(1)

    setup_logging(WorkerConfig.WORKER_ID)
    logging.info(f"Worker '{WorkerConfig.WORKER_ID}' khởi động.")
    logging.info(f"Server: {args.server_url}")
    logging.info(f"Videos dir: {videos_dir_path}")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Base URL: {args.base_url}")

    try:
        main_loop(args.server_url, videos_dir_path, args.mode, args.base_url)
    except KeyboardInterrupt:
        logging.info("Nhận Ctrl+C. Dừng worker...")
    finally:
        if WorkerConfig.WORKING_DIR.exists():
            logging.info(f"Dọn thư mục tạm: {WorkerConfig.WORKING_DIR}")
            shutil.rmtree(WorkerConfig.WORKING_DIR, ignore_errors=True)
        logging.info("Worker đã dừng.")
