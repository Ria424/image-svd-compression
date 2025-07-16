import os
from tkinter import filedialog

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from PIL import Image

# 한글 깨짐 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic" # 맑은 고딕
plt.rcParams["axes.unicode_minus"] = False

# 이미지 불러와 흑백 행렬로 변환
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

# 원본 크기 계산
original_bytes = A.size

# SVD 수행
U, S, VT = np.linalg.svd(A, full_matrices=False)

def compress(k: float):
    """특잇값 일부만 사용 (압축)

    k=사용할 상위 특잇값 수 (압축 수준 조절)
    """

    Σ_k = np.diag(S[:k])
    A_compressed = U[:, :k] @ Σ_k @ VT[:k, :]

    return A_compressed

# 이미지 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(A, cmap="gray")
axes[0].set_title(f"원본 이미지 ({original_bytes} bytes)")
axes[0].axis("off")

axes[1].set_title(f"압축 이미지")
axes[1].axis("off")

k_axes = fig.add_axes((0.1, 0.07, 0.8, 0.03,))

k_silder = Slider(
    ax=k_axes,
    label="k",
    valmin=10,
    valmax=100,
    valinit=80,
    valstep=10
)
k_silder.label.set_fontsize(12)
k_silder.valtext.set_fontsize(12)

compressed_img: np.typing.NDArray

def on_change(new_k: float) -> None:
    global compressed_img
    compressed_img = compress(new_k)

    # 압축 크기 계산
    compressed_bytes = U[:, :new_k].size + S[:new_k].size + VT[:new_k, :].size

    axes[1].imshow(compressed_img, cmap="gray")
    axes[1].set_title(f"압축 이미지 ({compressed_bytes} bytes)")

    # 압축률 출력
    compression_ratio = compressed_bytes / original_bytes

    plt.suptitle(
        f"압축률: {compression_ratio:.2%}",
        fontsize=12
    )

k_silder.on_changed(on_change)
on_change(k_silder.valinit)

btn_axes = fig.add_axes((0.4, 0.02, 0.2, 0.05,))

save_img_button = Button(
    ax=btn_axes,
    label="저장"
)

def on_clicked(_) -> None:
    # 압축된 이미지 저장
    save_path = f"compressed_image{img_ext}"
    Image.fromarray(compressed_img.astype(np.uint8)).save(save_path)

save_img_button.on_clicked(on_clicked)

plt.tight_layout()
plt.show()
