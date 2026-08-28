from ultralytics import YOLO
import cv2
import time
import subprocess  # 음성 출력을 위한 서브프로세스 라이브러리 추가


# Piper TTS 설정 상수
PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"
SPEAKER_DEVICE = "plughw:2,0"


# IoU(Intersection over Union)를 계산하는 함수
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# Jetson TensorRT 엔진(.engine) 포맷의 YOLO 모델 로드
model = YOLO("src/models/YOLO/yolo11n_int8.engine")

# GStreamer CSI 카메라 파이프라인
pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

displayed_fps = 0.0

# 5초 주차 유지 및 음성 중복 재생 방지를 위한 변수 초기화
parking_start_time = None
required_duration = 5.0
audio_played = False  # 음성이 이미 출력되었는지 기록하는 플래그

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_height, frame_width = frame.shape[:2]

    # 화면 중앙 1/9 크기의 주차 구역 설정
    zone_w = frame_width / 3
    zone_h = frame_height / 3
    zone_x1 = int((frame_width - zone_w) / 2)
    zone_y1 = int((frame_height - zone_h) / 2)
    zone_x2 = int(zone_x1 + zone_w)
    zone_y2 = int(zone_y1 + zone_h)
    parking_zone = [zone_x1, zone_y1, zone_x2, zone_y2]

    # YOLO 객체 감지
    results = model.predict(
        source=frame, conf=0.25, iou=0.5, verbose=False, classes=None
    )

    output_frame = results[0].plot()

    is_parked = False
    best_iou = 0.0
    current_in_zone = False

    for box in results[0].boxes:
        coords = box.xyxy[0].tolist()
        current_iou = calculate_iou(parking_zone, coords)

        if current_iou > best_iou:
            best_iou = current_iou

        if current_iou >= 0.6:
            current_in_zone = True

    elapsed_parking_time = 0.0

    if current_in_zone:
        if parking_start_time is None:
            parking_start_time = time.time()

        elapsed_parking_time = time.time() - parking_start_time

        if elapsed_parking_time >= required_duration:
            is_parked = True

            # [핵심] 5초를 채웠고, 아직 음성을 출력하지 않았다면 딱 한 번 실행
            if not audio_played:
                text = "주차 완료되었습니다!"
                try:
                    # 1. Piper로 텍스트를 음성 파일(.wav)로 변환
                    subprocess.run(
                        [
                            PIPER_PYTHON,
                            "-m",
                            "piper",
                            "-m",
                            PIPER_MODEL,
                            "-f",
                            OUTPUT_FILE,
                            "--",
                            text,
                        ],
                        check=True,
                    )
                    # 2. aplay를 통해 스피커로 출력
                    subprocess.run(
                        ["aplay", "-D", SPEAKER_DEVICE, OUTPUT_FILE],
                        check=True,
                    )
                except Exception as e:
                    print(f"Audio playback error: {e}")

                # 음성이 울렸으므로 플래그를 True로 변경 (중복 재생 방지)
                audio_played = True
    else:
        # 차가 주차 구역을 벗어나면 타이머와 음성 플래그를 모두 초기화 (다음 주차 시 다시 소리 나게 함)
        parking_start_time = None
        is_parked = False
        audio_played = False

    # 주차 구역 색상 지정 (성공 시 초록색)
    zone_color = (0, 255, 0) if is_parked else (255, 0, 0)
    cv2.rectangle(
        output_frame,
        (parking_zone[0], parking_zone[1]),
        (parking_zone[2], parking_zone[3]),
        zone_color,
        3,
    )

    # 상태 텍스트 설정
    if is_parked:
        status_text = f"Parking Status: SUCCESS! (Parked for 5s+)"
        status_color = (0, 255, 0)
    elif current_in_zone:
        remaining_time = max(0.0, required_duration - elapsed_parking_time)
        status_text = f"Parking... Hold steady! ({remaining_time:.1f}s left)"
        status_color = (0, 255, 255)
    else:
        status_text = f"Parking Status: Empty"
        status_color = (0, 0, 255)

    cv2.putText(
        output_frame,
        status_text,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    # FPS 계산 및 출력
    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    cv2.putText(
        output_frame,
        f"FPS: {displayed_fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLO Smart Parking System", output_frame)

cap.release()
cv2.destroyAllWindows()