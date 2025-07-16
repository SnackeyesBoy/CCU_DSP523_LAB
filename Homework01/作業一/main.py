import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------- 自訂函式區 ----------

#灰階
def grayscale(image):
    h, w, c = image.shape
    gray = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            y = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray[row, col] = y
    return gray

#邊緣偵測
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

#二值化
def binarize(gray_img, threshold=128):
    h, w = gray_img.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            binary[i, j] = 255 if gray_img[i, j] > threshold else 0
    return binary

#顯示結果
def display_pipeline(name, image, gray, edge, binary):
    # cv2.imshow(f"{name} - Original", image)
    # cv2.waitKey(0)
    # cv2.imshow(f"{name} - Grayscale", gray)
    # cv2.waitKey(0)
    # cv2.imshow(f"{name} - Edge Detection (ReLU)", edge)
    # cv2.waitKey(0)
    # cv2.imshow(f"{name} - Binarization", binary)
    # cv2.waitKey(0)
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image[:, :, ::-1])  # BGR 轉 RGB 
    plt.title(f"{name} - Original")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(edge, cmap='gray')
    plt.title("Edge Detection + ReLU")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(binary, cmap='gray')
    plt.title("Binarization")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ---------- 主程式 ----------

# 載入圖片
img1 = cv2.imread("作業一/liberty.png")
img2 = cv2.imread("作業一/temple.jpg")

# 顯示圖片資訊
print("Image 1 Shape:", img1.shape)
print("Image 2 Shape:", img2.shape)

# kernel
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
