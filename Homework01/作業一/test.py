import numpy as np
import cv2

# ---------- 自訂函式區 ----------

def grayscale(image):
    h, w, c = image.shape
    gray = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            y = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray[row, col] = y
    return gray

def convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

def relu(x):
    return np.maximum(0, x)

def binarize(gray_img, threshold=128):
    ret, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def display_pipeline(name, image, gray, edge, binary):
    cv2.imshow(f"{name} - Original", image)
    cv2.waitKey(0)
    cv2.imshow(f"{name} - Grayscale", gray)
    cv2.waitKey(0)
    cv2.imshow(f"{name} - Edge Detection (ReLU)", edge)
    cv2.waitKey(0)
    cv2.imshow(f"{name} - Binarization", binary)
    cv2.waitKey(0)

# ---------- 主程式 ----------

# 載入圖片
img1 = cv2.imread("liberty.png")
img2 = cv2.imread("temple.jpg")

# 顯示圖片資訊
print("Image 1 Shape:", img1.shape)
print("Image 2 Shape:", img2.shape)

# 定義邊緣偵測卷積核（Sobel X）
edge_kernel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

# 處理圖片 1
gray1 = grayscale(img1)
conv1 = convolve2d(gray1, edge_kernel)
relu1 = relu(conv1)
edge1 = np.uint8(np.clip(relu1, 0, 255))
binary1 = binarize(gray1)

# 處理圖片 2
gray2 = grayscale(img2)
conv2 = convolve2d(gray2, edge_kernel)
relu2 = relu(conv2)
edge2 = np.uint8(np.clip(relu2, 0, 255))
binary2 = binarize(gray2)

# 顯示結果
display_pipeline("Image 1", img1, gray1, edge1, binary1)
display_pipeline("Image 2", img2, gray2, edge2, binary2)

cv2.destroyAllWindows()
