import os
import time
import base64
import threading
import requests
import cv2
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()
app = Flask(__name__)

# ===== 환경 변수 =====
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "my-smart-recordings")
GCS_CREDENTIALS_FILE = "service-account-key.json"

VISION_API_KEY = os.getenv("VISION_API_KEY")
TRANSLATE_API_KEY = os.getenv("TRANSLATE_API_KEY")
VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
TRANSLATE_URL = f"https://translation.googleapis.com/language/translate/v2?key={TRANSLATE_API_KEY}"

# ===== 전역 상태 =====
latest_result = {}
latest_distance = None
recording = False
video_writer = None
recording_filename = ""
last_detect_time = 0

# ===== GCS 업로드 함수 =====
def upload_to_gcs(local_path, gcs_path):
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_FILE
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        blob.make_public()
        print(f"✅ GCS 업로드 완료: {blob.public_url}")
    except Exception as e:
        print("❌ GCS 업로드 실패:", e)

# ===== Vision API 분석 =====
def detect_labels_from_image(image_data):
    global latest_result
    try:
        b64_img = base64.b64encode(image_data).decode()
        payload = {
            "requests": [{
                "image": {"content": b64_img},
                "features": [{"type": "OBJECT_LOCALIZATION"}, {"type": "LABEL_DETECTION"}]
            }]
        }
        res = requests.post(VISION_URL, json=payload)
        response = res.json().get("responses", [{}])[0]
        objects = response.get("localizedObjectAnnotations", [])
        labels = response.get("labelAnnotations", [])
        best = objects[0]['name'] if objects else labels[0]['description'] if labels else "알 수 없음"
        trans = requests.post(TRANSLATE_URL, data={"q": best, "target": "ko"}).json()
        translated = trans['data']['translations'][0]['translatedText']
        latest_result = {"label_en": best, "label_ko": translated}
    except Exception as e:
        print("❌ Vision 분석 오류:", e)

# ===== 라우트 =====
@app.route('/')
def home():
    return "✅ ESP32 Render 서버 작동 중"

@app.route('/distance/update', methods=['POST'])
def update_distance():
    global latest_distance
    data = request.get_json()
    if not data or 'distance' not in data:
        return jsonify({"error": "distance 값 없음"}), 400
    latest_distance = data['distance']
    print(f"📏 거리 업데이트: {latest_distance}cm")
    return jsonify({"status": "ok"})

@app.route('/distance')
def get_distance():
    return jsonify({"distance_cm": latest_distance or "N/A"})

@app.route('/label')
def get_label():
    return jsonify(latest_result)

@app.route('/upload', methods=['POST'])
def upload_image():
    global recording, video_writer, recording_filename, last_detect_time
    img_bytes = request.data  # ESP32는 raw JPEG 전송
    if not img_bytes:
        return "이미지 없음", 400

    # 이미지 디코딩
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return "❌ 이미지 디코딩 실패", 400

    # 녹화 중이면 저장
    if recording and video_writer:
        video_writer.write(frame)

    # Vision 분석은 1초마다
    if time.time() - last_detect_time > 1:
        threading.Thread(target=detect_labels_from_image, args=(img_bytes,), daemon=True).start()
        last_detect_time = time.time()

    return "이미지 수신 완료", 200

@app.route('/record/start')
def start_record():
    global recording, video_writer, recording_filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    recording_filename = f"record_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(recording_filename, fourcc, 20.0, (640, 480))
    recording = True
    print(f"▶️ 녹화 시작: {recording_filename}")
    return "녹화 시작"

@app.route('/record/stop')
def stop_record():
    global recording, video_writer, recording_filename
    recording = False
    if video_writer:
        video_writer.release()
        video_writer = None
        print(f"⏹ 녹화 종료: {recording_filename}")
        threading.Thread(
            target=upload_to_gcs,
            args=(recording_filename, f"recordings/{recording_filename}"),
            daemon=True
        ).start()
    return "녹화 종료"

@app.route('/videos')
def list_videos():
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_FILE
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix="recordings/")
        return jsonify({
            "videos": [
                {"name": b.name.split('/')[-1], "url": b.public_url}
                for b in blobs if b.name.endswith(".mp4")
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 서버 실행 =====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
