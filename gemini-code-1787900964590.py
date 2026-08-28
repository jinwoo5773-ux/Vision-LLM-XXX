import time
import cv2
from ultralytics import YOLO

model = YOLO("src/models/YOLO/yolo11n_int8.engine")

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


# 두 바운딩 박스의 IoU(Intersection over Union)를 계산하는 함수
def calculate_iou(box1, box2):
  # box 형식: [xmin, ymin, xmax, ymax]
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])

  # 겹치는 영역(교집합)의 넓이 계산
  inter_area = max(0, x2 - x1) * max(0, y2 - y1)
  if inter_area == 0:
    return 0.0

  # 각각의 바운딩 박스 넓이 계산
  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

  # 합집합 넓이 계산 (합집합 = box1 넓이 + box2 넓이 - 교집합 넓이)
  union_area = box1_area + box2_area - inter_area

  # IoU = 교집합 / 합집합
  return inter_area / union_area if union_area > 0 else 0.0


displayed_fps = 0.0

while True:
  start_time = time.perf_counter()

  ret, frame = cap.read()
  if not ret:
    break

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

  # 현재 프레임의 해상도(너비, 높이) 가져오기 (1280x720)
  height, width = frame.shape[:2]

  # 1. 화면 중앙에 전체 화면 면적의 1/9 크기를 가진 주차 구역 설정
  # 가로와 세로를 각각 3등분 하므로, 면적은 (1/3) * (1/3) = 1/9이 됩니다.
  parking_box = [
      width / 3,  # xmin (왼쪽 1/3 지점)
      height / 3,  # ymin (위쪽 1/3 지점)
      width * 2 / 3,  # xmax (오른쪽 2/3 지점)
      height * 2 / 3,  # ymax (아래쪽 2/3 지점)
  ]

  results = model.predict(
      source=frame,
      conf=0.25,
      iou=0.5,
      verbose=False,
      classes=None,
  )

  # YOLO 모델이 그려준 기본 바운딩 박스가 포함된 프레임 가져오기
  output_frame = results[0].plot()

  # 주차 상태 판정을 위한 변수 초기화
  is_parked = False
  max_iou = 0.0
  iou_threshold = 0.6  # 주차 완료로 인정할 IoU 기준치 (상황에 맞게 조절 가능)

  # 감지된 객체들의 바운딩 박스 좌표 추출
  if results[0].boxes is not None:
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [xmin, ymin, xmax, ymax] 좌표들

    for box in boxes:
      # 고정된 주차 구역(parking_box)과 감지된 객체 박스 간의 IoU 계산
      current_iou = calculate_iou(parking_box, box)

      # 가장 많이 겹친 객체의 IoU 값을 추적
      if current_iou > max_iou:
        max_iou = current_iou

      # IoU가 기준치를 넘으면 정확히 주차된 것으로 판단
      if current_iou >= iou_threshold:
        is_parked = True

  # 2. 고정 주차 구역 박스를 화면에 직접 시각화
  p_xmin, p_ymin, p_xmax, p_ymax = (
      int(parking_box[0]),
      int(parking_box[1]),
      int(parking_box[2]),
      int(parking_box[3]),
  )

  # 주차가 완료되면 초록색, 비어있으면 파란색 테두리로 표시
  zone_color = (0, 255, 0) if is_parked else (255, 0, 0)
  cv2.rectangle(output_frame, (p_xmin, p_ymin), (p_xmax, p_ymax), zone_color, 3)
  cv2.putText(
      output_frame,
      f"Parking Zone (IoU: {max_iou:.2f})",
      (p_xmin, p_ymin - 10),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      zone_color,
      2,
  )

  # 3. 화면 상단에 주차 성공 여부 텍스트 출력
  status_text = (
      "Status: PARKED (Success!)" if is_parked else "Status: EMPTY / PARKING..."
  )
  status_color = (0, 255, 0) if is_parked else (0, 0, 255)
  cv2.putText(
      output_frame,
      status_text,
      (20, 80),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      status_color,
      2,
  )

  # FPS 계산 및 출력
  elapsed_time = time.perf_counter() - start_time
  current_fps = 1.0 / elapsed_time

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

  cv2.imshow("YOLO Object Detection with FPS", output_frame)

cap.release()
cv2.destroyAllWindows()