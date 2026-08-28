from ultralytics import YOLO
import cv2
import time


# IoU(Intersection over Union)를 계산하는 함수
def calculate_iou(box1, box2):
    # box1, box2 형식: [x1, y1, x2, y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 겹치는 영역의 넓이 계산
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # 각 박스의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union(합집합) 넓이 계산 (합집합 = box1 + box2 - 교집합)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# YOLO 모델 로드 (Jetson 최적화된 TensorRT .engine 모델)
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

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 현재 프레임의 해상도 가져오기 (1280x720)
    frame_height, frame_width = frame.shape[:2]

    # [핵심] 화면 중앙에 전체 넓이의 1/9 크기를 가지는 주차 구역 바운딩 박스 설정
    # 넓이가 1/9이 되려면 가로와 세로 각각 1/3 크기여야 합니다 (1/3 * 1/3 = 1/9)
    zone_w = frame_width / 3
    zone_h = frame_height / 3
    zone_x1 = int((frame_width - zone_w) / 2)
    zone_y1 = int((frame_height - zone_h) / 2)
    zone_x2 = int(zone_x1 + zone_w)
    zone_y2 = int(zone_y1 + zone_h)
    parking_zone = [zone_x1, zone_y1, zone_x2, zone_y2]

    # YOLO 모델 예측 수행
    results = model.predict(
        source=frame,
        conf=0.25,  # 신뢰도 임계값
        iou=0.5,  # NMS IoU 임계값
        verbose=False,
        classes=None,  # 필요시 특정 클래스만 지정 (예: 차는 보통 class 2)
    )

    output_frame = results[0].plot()

    # 주차 완료 상태를 판별하기 위한 변수
    is_parked = False
    best_iou = 0.0

    # 감지된 객체들의 바운딩 박스와 주차 구역 간의 IoU 비교
    for box in results[0].boxes:
        # 박스 좌표 추출 [x1, y1, x2, y2]
        coords = box.xyxy[0].tolist()

        # 주차 구역과 감지된 객체 박스 간의 IoU 계산
        current_iou = calculate_iou(parking_zone, coords)

        if current_iou > best_iou:
            best_iou = current_iou

        # IoU가 0.6 이상(설정에 따라 조절 가능)으로 겹치면 주차 완료로 판단
        if current_iou >= 0.6:
            is_parked = True

    # 주차 구역 박스 색상 결정 (주차 성공 시 초록색, 평소엔 파란색)
    zone_color = (0, 255, 0) if is_parked else (255, 0, 0)
    # 화면에 중앙 주차 구역 박스 그리기
    cv2.rectangle(
        output_frame,
        (parking_zone[0], parking_zone[1]),
        (parking_zone[2], parking_zone[3]),
        zone_color,
        3,
    )

    # 주차 상태 텍스트 화면에 표시
    status_text = (
        f"Parking Status: {'SUCCESS (Parked!)' if is_parked else 'Empty'} (IoU: {best_iou:.2f})"
    )
    status_color = (0, 255, 0) if is_parked else (0, 0, 255)
    cv2.putText(
        output_frame,
        status_text,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
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