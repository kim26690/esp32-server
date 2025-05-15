import os, cv2, time, base64, threading, requests
from flask import Flask, Response, jsonify
from google.cloud import storage

app = Flask(__name__)

# ===== 설정 =====
ESP32_IP = os.getenv("ESP32_IP", "192.168.0.100")  # 기본값 지정
RTSP_URL = f"rtsp://{ESP32_IP}:8554/mjpeg/1"
DISTANCE_URL = f"http://{ESP32_IP}/distance"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "my-smart-recordings")
GCS_CREDENTIALS_FILE = "service-account-key.json"  # Render에서는 이 파일을 프로젝트 루트에 배치

# RTSP 연결 확인
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise RuntimeError(f"❌ RTSP 스트림 연결 실패: {RTSP_URL}")
print("✅ RTSP 연결 성공")

# API 키 (환경변수에서 불러오기)
VISION_API_KEY = os.getenv("VISION_API_KEY")
TRANSLATE_API_KEY = os.getenv("TRANSLATE_API_KEY")
VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={VISION_API_KEY}"
TRANSLATE_URL = f"https://translation.googleapis.com/language/translate/v2?key={TRANSLATE_API_KEY}"

# ===== 전역 변수 =====
latest_result = {}
latest_distance = None
last_detect_time = 0
recording = False
video_writer = None
recording_filename = ""

# ===== GCS 업로드 함수 =====
def upload_to_gcs(local_path, gcs_path):
    try:
        if not os.path.exists(GCS_CREDENTIALS_FILE):
            print("❌ 인증 파일 없음:", GCS_CREDENTIALS_FILE)
            return
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_FILE
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        try:
            blob.make_public()
        except Exception as e:
            print("⚠️ 공개 설정 실패 (무시 가능):", e)
        print(f"✅ GCS 업로드 완료: {blob.public_url}")
    except Exception as e:
        print("❌ GCS 업로드 실패:", e)

# ===== Vision 분석 =====
def detect_labels(frame):
    global latest_result
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        b64_img = base64.b64encode(buffer).decode()
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

# ===== 거리 측정 =====
def update_distance():
    global latest_distance
    while True:
        try:
            r = requests.get(DISTANCE_URL, timeout=1)
            if r.status_code == 200:
                latest_distance = int(r.text.strip())
        except:
            latest_distance = None
        time.sleep(1)

# ===== 영상 스트리밍 =====
def generate_frames():
    global last_detect_time, recording, video_writer
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ 프레임 읽기 실패")
            time.sleep(0.1)
            continue
        if recording and video_writer:
            video_writer.write(frame)
        if time.time() - last_detect_time > 1:
            threading.Thread(target=detect_labels, args=(frame,), daemon=True).start()
            last_detect_time = time.time()
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ===== API 라우터 =====
@app.route('/')
def index():
    return "📡 ESP32 Vision 서버 정상 동작 중"

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/distance')
def distance():
    print("📏 현재 거리 값:", latest_distance)
    return jsonify({'distance_cm': latest_distance if latest_distance is not None else "N/A"})

@app.route('/label')
def label():
    return jsonify(latest_result)

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
def videos():
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_FILE
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blobs = bucket.list_blobs(prefix="recordings/")
        return jsonify({
            "videos": [
                {"name": b.name.split('/')[-1], "url": b.public_url}
                for b in blobs if b.name.endswith(".mp4") and b.public_url.startswith("http")
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 실행 =====
if __name__ == '__main__':
    threading.Thread(target=update_distance, daemon=True).start()
    app.run(host='0.0.0.0', port=8000)
