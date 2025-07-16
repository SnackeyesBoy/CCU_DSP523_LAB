import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def mean_filter(Image, kernel_size=3):
    pad = kernel_size // 2 
    h, w = Image.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # 邊緣補 0（zero padding）
    padded_img = np.pad(Image, pad_width=pad, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            # 取出區域塊
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            #print(region)
            # 計算平均值
            mean_value = np.mean(region)
            result[i, j] = int(mean_value)

    return result

def median_filter(Image, kernel_size=3):
    pad = kernel_size // 2 
    h, w = Image.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Zero padding
    padded_img = np.pad(Image, pad_width=pad, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            # 取出 kernel 區域並展平成一維
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            flat = region.flatten()
            sorted_data = sorted(flat)
            n = len(sorted_data)
            median_value = sorted_data[n // 2]  # 中位數
            result[i, j] = int(median_value)

    return result

def display(name, img, gray, mean, median):
    plt.figure(figsize=(16, 8))  
    
    # 原圖
    plt.subplot(2, 4, 1)
    plt.imshow(img[:, :, ::-1])  # BGR to RGB
    plt.title(f"{name} - Original")
    plt.axis('off')

    # 分開畫 R/G/B 通道
    plt.subplot(2, 4, 2)
    b, g, r = cv2.split(img)
    plt.hist(b.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.5, label='B')
    plt.hist(g.ravel(), bins=256, range=(0, 256), color='green', alpha=0.5, label='G')
    plt.hist(r.ravel(), bins=256, range=(0, 256), color='red', alpha=0.5, label='R')
    plt.title(f"Histogram - {name} (RGB Channels)")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.legend()

    # Grayscale
    plt.subplot(2, 4, 3)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.hist(gray.ravel(), bins=256, range=(0, 255), color='gray')
    plt.title("Histogram - Grayscale")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])

    # Mean filter
    plt.subplot(2, 4, 5)
    plt.imshow(mean, cmap='gray')
    plt.title("Mean Filter")
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.hist(mean.ravel(), bins=256, range=(0, 255), color='skyblue')
    plt.title("Histogram - Mean Filter")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])

    # Median filter
    plt.subplot(2, 4, 7)
    plt.imshow(median, cmap='gray')
    plt.title("Median Filter")
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.hist(median.ravel(), bins=256, range=(0, 255), color='green')
    plt.title("Histogram - Median Filter")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])

    plt.tight_layout()
    plt.show()

# ---------- 主程式 ----------
#圖片路徑
img = cv2.imread("作業二/noise_image.png")

gray_img = grayscale(img) #灰階
mean_img = mean_filter(gray_img) #均值濾波
median_img = median_filter(gray_img) #中值濾波
print("Image 1 Original Shape:", img.shape)
print("Image 1 gray_img Shape:", gray_img.shape)
print("Image 1 mean_img Shape:", mean_img.shape)
print("Image 1 median_img Shape:", median_img.shape)

display("noise_image.png" , img , gray_img , mean_img , median_img)

# cv2.imshow("Original Image",img)
# cv2.waitKey(0)
# cv2.imshow("Mean Filter",mean_img)
# cv2.waitKey(0)
# cv2.imshow("Median Filter",median_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

