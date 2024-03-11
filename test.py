# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import numpy as np
import cv2
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint


torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

print("*" * 20 + " Testing zoedepth " + "*" * 20)
conf = get_config("zoedepth", "infer")


print("Config:")
pprint(conf)

model = build_model(conf).to(DEVICE)
model.eval()
x = torch.rand(1, 3, 384, 512).to(DEVICE)

print("-"*20 + "Testing on a random input" + "-"*20)

with torch.no_grad():
    out = model(x)

if isinstance(out, dict):
    # print shapes of all outputs
    for k, v in out.items():
        if v is not None:
            print(k, v.shape)
else:
    print([o.shape for o in out if o is not None])

# Test img

# 웹캠 초기화
cap = cv2.VideoCapture(0)

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
        pred = Image.fromarray(colorize(out.squeeze()))

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

