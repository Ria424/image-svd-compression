import os
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# 한글 깨짐 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic" # 맑은 고딕
plt.rcParams["axes.unicode_minus"] = False

# 1. 이미지 불러와 흑백 행렬로 변환
img_path = filedialog.askopenfilename(
    filetypes=(
        ("All File", "*.*",),
        ("PNG File", "*.png",),
        ("JPEG File", ("*.jpg", "*.jpeg",),),
        ("WEBP File", ("*.webp",),),
    )
)
if not img_path:
    raise Exception("이미지를 선택해주세요.")

# 불러온 이미지의 확장자 저장
# 나중에 압축된 이미지를 저장할 때 쓰임
img_ext = os.path.splitext(img_path)[1]

# 흑백 이미지로 바꾸기
img = Image.open(img_path).convert("L")

# 이미지를 행렬로 나타내기
A = np.array(img)

# 2. 원본 크기 계산
original_bytes = A.size

# 3. SVD 수행
U, S, VT = np.linalg.svd(A, full_matrices=False)

# 4. 특잇값 일부만 사용 (압축)
k = 50  # 압축 수준 조절
Σ_k = np.diag(S[:k])
A_compressed = U[:, :k] @ Σ_k @ VT[:k, :]

# 5. 압축 크기 계산
compressed_bytes = U[:, :k].size + S[:k].size + VT[:k, :].size

# 6. 압축률 출력
compression_ratio = compressed_bytes / original_bytes
print(f"압축 전 크기: {original_bytes} bytes")
print(f"압축 후 크기: {compressed_bytes} bytes")
print(f"압축률: {compression_ratio:.2%} (k={k})")

# 7. 압축된 이미지 저장
save_path = f"compressed_image{img_ext}"
Image.fromarray(A_compressed.astype(np.uint8)).save(save_path)

# 8. 이미지 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(A, cmap="gray")
axes[0].set_title("원본 이미지")
axes[0].axis("off")

axes[1].imshow(A_compressed, cmap="gray")
axes[1].set_title(f"압축 이미지 (k={k})")
axes[1].axis("off")

plt.suptitle(
    f"압축률: {compression_ratio:.2%} - 저장 경로: {os.path.abspath(save_path)}",
    fontsize=12
)
plt.tight_layout()
plt.show()