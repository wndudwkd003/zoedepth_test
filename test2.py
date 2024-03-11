import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize

# ZoeDepth 모델 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
conf = get_config("zoedepth", "infer")
model = build_model(conf).to(DEVICE)
model.eval()

# 웹캠 초기화
cap = cv2.VideoCapture(0)

depth_map = None  # 깊이 맵을 저장할 변수

# 마우스 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    global depth_map
    if event == cv2.EVENT_LBUTTONDBLCLK and depth_map is not None:
        # 더블 클릭한 위치의 깊이 값을 추출
        depth_value = depth_map[y, x]
        print(f"Depth at ({x}, {y}): {depth_value:.2f} units")
        cv2.putText(frame, f"{depth_value:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# OpenCV 윈도우에 마우스 콜백 함수 설정
cv2.namedWindow("Depth Estimation")
cv2.setMouseCallback("Depth Estimation", mouse_callback)

try:
    while True:
        # 웹캠으로부터 이미지 캡처
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # OpenCV 이미지를 PIL 이미지로 변환
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 모델 입력 사이즈로 이미지 조정
        img_resized = img.resize((512, 384), Image.ANTIALIAS)

        # 모델 입력을 위해 이미지 전처리
        X = ToTensor()(img_resized).unsqueeze(0).to(DEVICE)

        # 모델을 통해 심도 추정
        with torch.no_grad():
            out = model.infer(X).cpu()

        # 결과 이미지 생성
        depth_map = out.squeeze()
        pred = Image.fromarray(colorize(depth_map))

        # 결과 이미지를 OpenCV 형식으로 변환
        pred_cv = cv2.cvtColor(np.array(pred), cv2.COLOR_RGB2BGR)

        # 결과 이미지 표시
        cv2.imshow('Depth Estimation', pred_cv)

        # 'q'를 누르면 반복문 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
