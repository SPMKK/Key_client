# worker.py
import argparse
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Literal

import requests

# Import lớp xử lý chính từ mã nguồn của bạn
# Đảm bảo file 'infer_concurrent.py' nằm trong cùng thư mục hoặc trong PYTHONPATH
try:
    from infer_concurent import VideoKeyframeExtractor
except ImportError:
    print("Lỗi: Không thể import 'VideoKeyframeExtractor' từ 'infer_concurrent.py'.")
    print("Hãy đảm bảo 'worker.py' và 'infer_concurrent.py' ở trong cùng một thư mục.")
    exit(1)


# --- 1. Cấu hình Worker ---
class WorkerConfig:
    """Cấu hình cho worker xử lý keyframe."""
    # ID của worker, hữu ích cho việc logging khi chạy nhiều worker
    WORKER_ID: str = f"worker-{os.getpid()}"

    # Thư mục làm việc tạm thời cho mỗi task
    WORKING_DIR: Path = Path("./worker_temp")

    # Các tham số cho việc trích xuất keyframe (giống như trong infer_concurrent.py)
    SAMPLE_RATE: int = 5
    MAX_FRAMES_PER_SHOT: int = 55

    # Khoảng thời gian nghỉ (giây) giữa các lần hỏi server nếu không có task
    POLL_INTERVAL_SECONDS: int = 10


# --- 2. Cấu hình Logging ---
def setup_logging(worker_id: str):
    """Thiết lập logging để ghi ra file và console."""
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
    # Tắt logging quá chi tiết từ thư viện requests
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- 3. Các hàm tương tác với Server ---

def get_task_from_server(server_url: str, mode: str) -> Optional[Dict]:
    """
    Lấy một task mới từ server.
    """
    get_batch_url = f"{server_url}/get_batch"
    # Thêm 'mode' vào payload để server biết cách phản hồi
    payload = {"size": 1, "mode": mode} 
    try:
        logging.info(f"Đang hỏi server {get_batch_url} để lấy task mới (mode={mode})...")
        response = requests.post(get_batch_url, json=payload, timeout=15)
        response.raise_for_status()

        tasks = response.json()
        if not tasks:
            logging.info("Không có task nào đang chờ. Sẽ thử lại sau.")
            return None

        task = tasks[0]
        logging.info(f"Đã nhận task: {task}") # Log toàn bộ task để thấy result_path
        return task
    except requests.exceptions.RequestException as e:
        logging.error(f"Không thể kết nối hoặc giao tiếp với server: {e}")
        return None
    except Exception as e:
        logging.error(f"Lỗi không xác định khi lấy task: {e}")
        return None

#----LOCAL ONLY ----
def report_completion_to_server(server_url: str, video_id: str) -> bool:
    """Chỉ sử dụng ở mode 'local'."""
    report_url = f"{server_url}/report_done"
    try:
        logging.info(f"Đang báo cáo hoàn thành cho server về video_id '{video_id}'...")
        # Sử dụng params thay vì data cho phương thức GET/POST với query string
        response = requests.post(report_url, params={'video_id': video_id}, timeout=30)
        response.raise_for_status()
        logging.info(f"Server đã xác nhận hoàn thành cho '{video_id}'. Phản hồi: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Báo cáo hoàn thành cho '{video_id}' thất bại: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Chi tiết lỗi từ server: {e.response.text}")
        return False
    except Exception as e:
        logging.error(f"Lỗi không xác định khi báo cáo hoàn thành: {e}")
        return False
    

def upload_result_to_server(server_url: str, video_id: str, zip_path: Path) -> bool:
    """
    Tải file ZIP kết quả lên server.
    Returns:
        True nếu upload thành công, ngược lại là False.
    """
    upload_url = f"{server_url}/upload_result"
    try:
        logging.info(f"Đang tải kết quả cho video_id '{video_id}' từ file {zip_path}...")
        with open(zip_path, 'rb') as f:
            files = {'file': (zip_path.name, f, 'application/zip')}
            data = {'video_id': video_id}
            response = requests.post(upload_url, files=files, data=data, timeout=300) # Timeout 5 phút
            response.raise_for_status()

        logging.info(f"Upload thành công cho video_id '{video_id}'. Phản hồi từ server: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Upload thất bại cho video_id '{video_id}': {e}")
        if e.response is not None:
            logging.error(f"Chi tiết lỗi từ server: {e.response.text}")
        return False
    except Exception as e:
        logging.error(f"Lỗi không xác định khi upload kết quả: {e}")
        return False

# --- 4. Các hàm xử lý file ---

def create_result_zip(output_dir: Path, video_id: str, zip_destination_dir: Path) -> Optional[Path]:
    """
    Tạo một file ZIP chứa kết quả xử lý.
    File ZIP sẽ chứa 'metadata.json' và thư mục 'keyframes/'.
    """
    source_dir = output_dir / video_id
    zip_path = zip_destination_dir / f"{video_id}_result.zip"

    # Kiểm tra các thành phần cần thiết
    metadata_file = source_dir / "metadata.json"
    keyframes_dir = source_dir / "keyframes"

    if not metadata_file.exists() or not keyframes_dir.exists() or not keyframes_dir.is_dir():
        logging.error(f"Kết quả xử lý cho '{video_id}' không đầy đủ. Thiếu metadata.json hoặc thư mục keyframes/.")
        return None

    logging.info(f"Đang tạo file ZIP từ {source_dir}...")
    try:
        # Sử dụng shutil.make_archive để tạo file zip một cách an toàn
        # Nó sẽ tạo một file zip chứa nội dung của source_dir
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(source_dir))
        logging.info(f"Đã tạo file ZIP thành công: {zip_path}")
        return zip_path
    except Exception as e:
        logging.error(f"Không thể tạo file ZIP: {e}")
        return None


# --- 5. Vòng lặp xử lý chính ---

def main_loop(server_url: str, videos_dir: Path, mode: Literal['local', 'colab']):
    """
    Vòng lặp chính của worker: lấy task, xử lý, và gửi kết quả.
    """
    logging.info(f"Worker đang hoạt động ở chế độ: {mode.upper()}")
    
    # --- PHẦN BỊ THIẾU ĐÃ ĐƯỢC THÊM LẠI ---
    # Khởi tạo một lần duy nhất
    worker_temp_dir = WorkerConfig.WORKING_DIR # Dùng cho mode colab
    worker_temp_dir.mkdir(exist_ok=True)

    logging.info("Đang khởi tạo VideoKeyframeExtractor...")
    try:
        # Khởi tạo extractor. Thư mục output sẽ được ghi đè sau,
        # nên giá trị ban đầu không quá quan trọng.
        extractor = VideoKeyframeExtractor(
            transnet_weights="transnetv2-weights",
            output_dir=str(worker_temp_dir), # Mặc định là thư mục tạm
            sample_rate=WorkerConfig.SAMPLE_RATE,
            max_frames_per_shot=WorkerConfig.MAX_FRAMES_PER_SHOT
        )
        logging.info("VideoKeyframeExtractor đã sẵn sàng.")
    except Exception as e:
        logging.critical(f"Không thể khởi tạo VideoKeyframeExtractor: {e}. Worker sẽ thoát.", exc_info=True)
        return
    # --- KẾT THÚC PHẦN SỬA LỖI ---

    while True:
        task = get_task_from_server(server_url, mode)

        if not task:
            time.sleep(WorkerConfig.POLL_INTERVAL_SECONDS)
            continue

        video_id = task['video_id']
        source_video_path = videos_dir / task['filename']
        task_output_dir = None # Để dọn dẹp
        zip_file_path = None

        try:
            if not source_video_path.exists():
                logging.error(f"Không tìm thấy file video nguồn: {source_video_path}. Bỏ qua task.")
                continue

            logging.info(f"Bắt đầu xử lý video: {source_video_path}")
            start_time = time.time()

            # ----- LOGIC THEO MODE -----
            if mode == 'local':
                # Server cung cấp đường dẫn kết quả tuyệt đối
                result_path = Path(task['result_path'])
                result_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Ghi đè thư mục output của extractor cho task này
                extractor.output_dir = str(result_path.parent)
                extractor.extract_keyframes(str(source_video_path))

                logging.info(f"Hoàn thành xử lý '{video_id}' trong {time.time() - start_time:.2f} giây.")

                # Báo cáo hoàn thành cho server, không cần upload
                report_completion_to_server(server_url, video_id)

            else: # mode == 'colab' (logic cũ)
                # Ghi vào thư mục tạm thời của worker
                task_output_dir = worker_temp_dir / video_id
                extractor.output_dir = str(worker_temp_dir)
                extractor.extract_keyframes(str(source_video_path))
                
                logging.info(f"Hoàn thành xử lý '{video_id}' trong {time.time() - start_time:.2f} giây.")

                # Nén và tải kết quả lên server
                zip_file_path = create_result_zip(worker_temp_dir, video_id, worker_temp_dir)
                if zip_file_path:
                    upload_result_to_server(server_url, video_id, zip_file_path)
        
        except Exception as e:
            logging.error(f"Lỗi nghiêm trọng xảy ra khi xử lý task '{video_id}': {e}", exc_info=True)

        finally:
            logging.info(f"Bắt đầu dọn dẹp cho task '{video_id}'...")
            if zip_file_path and zip_file_path.exists():
                os.remove(zip_file_path)
            
            # Chỉ xóa thư mục output nếu là mode colab (vì nó là thư mục tạm)
            if mode == 'colab' and task_output_dir and task_output_dir.exists():
                try:
                    shutil.rmtree(task_output_dir)
                    logging.info(f"Đã xóa thư mục kết quả tạm thời: {task_output_dir}")
                except OSError as e:
                    logging.warning(f"Không thể xóa thư mục tạm {task_output_dir}: {e}")
            
            logging.info(f"Hoàn tất dọn dẹp cho task '{video_id}'.")
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Worker để xử lý video, trích xuất keyframe và gửi kết quả về server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:1202",
        help="URL của Task Dispatcher Server."
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        required=True,
        help="Đường dẫn đến thư mục 'videos' được chia sẻ, nơi chứa các file video gốc."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['local', 'colab'],
        default='colab',
        help="Chế độ hoạt động: 'local' để ghi trực tiếp vào thư mục kết quả, 'colab' để nén và upload."
    )
    args = parser.parse_args()

    # Chuyển đổi đường dẫn thành đối tượng Path
    videos_dir_path = Path(args.videos_dir)
    if not videos_dir_path.exists() or not videos_dir_path.is_dir():
        print(f"Lỗi: Thư mục videos '{videos_dir_path}' không tồn tại hoặc không phải là một thư mục.")
        exit(1)

    setup_logging(WorkerConfig.WORKER_ID)
    logging.info(f"Worker '{WorkerConfig.WORKER_ID}' đang bắt đầu...")
    logging.info(f"Kết nối tới server tại: {args.server_url}")
    logging.info(f"Sử dụng thư mục videos tại: {videos_dir_path}")

    try:
        main_loop(args.server_url, videos_dir_path, args.mode)
    except KeyboardInterrupt:
        logging.info("Đã nhận tín hiệu thoát (Ctrl+C). Worker đang dừng...")
    finally:
        # Dọn dẹp thư mục làm việc chính khi thoát
        if WorkerConfig.WORKING_DIR.exists():
            logging.info(f"Đang dọn dẹp thư mục làm việc chính: {WorkerConfig.WORKING_DIR}")
            shutil.rmtree(WorkerConfig.WORKING_DIR, ignore_errors=True)
        logging.info("Worker đã dừng.")
