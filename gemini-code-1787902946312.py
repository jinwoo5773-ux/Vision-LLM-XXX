from ultralytics import YOLO  # YOLO 모델을 사용하기 위한 라이브러리 임포트
import cv2  # 영상 처리 및 화면 출력을 위한 OpenCV 라이브러리 임포트
import time  # 시간 측정 및 FPS 계산을 위한 time 라이브러리 임포트


# 두 개의 바운딩 박스 간의 IoU(Intersection over Union)를 계산하는 함수
def calculate_iou(box1, box2):
    # box1, box2 형식: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 겹치는 영역의 가로, 세로 길이 계산
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # 각 바운딩 박스의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 합집합(Union) 넓이 계산
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# Jetson TensorRT 엔진(.engine) 포맷의 YOLO 모델 로드
model = YOLO("src/models/YOLO/yolo11n_int8.engine")

# GStreamer를 통한 Jetson CSI 카메라 파이프라인 설정
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

# [추가] 5초 주차 유지를 위한 타이머 변수 초기화
parking_start_time = None
required_duration = 5.0  # 성공으로 인정할 유지 시간 (초)

while True:
    start_time = time.perf_counter()  # 현재 프레임 처리 시작 시간 기록

    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_height, frame_width = frame.shape[:2]

    # 화면 중앙에 전체 넓이의 1/9 크기를 가지는 주차 구역 바운딩 박스 계산
    zone_w = frame_width / 3
    zone_h = frame_height / 3
    zone_x1 = int((frame_width - zone_w) / 2)
    zone_y1 = int((frame_height - zone_h) / 2)
    zone_x2 = int(zone_x1 + zone_w)
    zone_y2 = int(zone_y1 + zone_h)
    parking_zone = [zone_x1, zone_y1, zone_x2, zone_y2]

    # YOLO 객체 감지 수행
    results = model.predict(
        source=frame, conf=0.25, iou=0.5, verbose=False, classes=None
    )

    output_frame = results[0].plot()

    is_parked = False
    best_iou = 0.0
    current_in_zone = False  # 현재 프레임에서 구역 안에 차가 들어와 있는지 여부

    # 감지된 객체들을 순회하며 주차 구역과의 겹침 확인
    for box in results[0].boxes:
        coords = box.xyxy[0].tolist()
        current_iou = calculate_iou(parking_zone, coords)

        if current_iou > best_iou:
            best_iou = current_iou

        # IoU가 0.6 이상 겹치면 주차 구역 안에 들어온 것으로 판단
        if current_iou >= 0.6:
            current_in_zone = True

    # [핵심 로직] 5초 지속 시간 체크
    elapsed_parking_time = 0.0

    if current_in_zone:
        if parking_start_time is None:
            # 처음 주차 구역에 진입한 순간의 시간 기록
            parking_start_time = time.time()

        # 진입한 이후부터 흘러간 시간 계산
        elapsed_parking_time = time.time() - parking_start_time

        # 5초 이상 유지되었다면 최종 SUCCESS 처리
        if elapsed_parking_time >= required_duration:
            is_parked = True
    else:
        # 차가 구역을 벗어나거나 없으면 타이머 리셋
        parking_start_time = None
        is_parked = False

    # 주차 구역 테두리 색상 결정 (성공: 초록색, 유지 중/비어있음: 파란색 또는 빨간색)
    zone_color = (0, 255, 0) if is_parked else (255, 0, 0)
    cv2.rectangle(
        output_frame,
        (parking_zone[0], parking_zone[1]),
        (parking_zone[2], parking_zone[3]),
        zone_color,
        3,
    )

    # 주차 상태 및 5초 카운트다운 텍스트 설정
    if is_parked:
        status_text = f"Parking Status: SUCCESS! (Parked for 5s+)"
        status_color = (0, 255, 0)  # 초록색
    elif current_in_zone:
        remaining_time = max(0.0, required_duration - elapsed_parking_time)
        status_text = f"Parking... Hold steady! ({remaining_time:.1f}s left)"
        status_color = (0, 255, 255)  # 노란색 (유지 중)
    else:
        status_text = f"Parking Status: Empty"
        status_color = (0, 0, 255)  # 빨간색 (비어있음)

    cv2.putText(
        output_frame,
        status_text,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
    )

    # FPS 계산 및 화면 출력
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