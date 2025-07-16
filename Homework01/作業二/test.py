import cv2
import numpy as np

def mean_filter(img, kernel_size=3):
    pad = kernel_size // 2
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # 邊緣補 0（zero padding）
    padded_img = np.pad(img, pad_width=pad, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            # 取出區域塊
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            # 計算平均值
            mean_value = np.mean(region)
            result[i, j] = int(mean_value)

    return result

# 測試程式
img = cv2.imread('noise_image.png', cv2.IMREAD_GRAYSCALE)  # 灰階讀入
filtered = mean_filter(img, kernel_size=3)

cv2.imshow("Original", img)
cv2.imshow("Mean Filtered", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
