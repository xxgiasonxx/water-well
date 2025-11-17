import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

def get_dark_channel(img, patch_size=15):
    """
    計算圖像的暗通道 (Dark Channel)。

    Args:
        img (numpy.ndarray): BGR 格式的圖像 (0-255)。
        patch_size (int): 進行局部最小化的窗口大小。

    Returns:
        numpy.ndarray: 暗通道圖像 (H x W)。
    """
    # 1. 找到每個像素在 RGB 三個通道中的最小值
    min_channel = np.min(img, axis=2) # 得到 H x W 的結果

    # 2. 在 min_channel 上進行局部最小化 (使用 min filter)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel

def get_atmospheric_light(img, dark_channel, top_percent=0.001):
    """
    估算大氣光 (Atmospheric Light, A)。

    Args:
        img (numpy.ndarray): 原始圖像。
        dark_channel (numpy.ndarray): 暗通道圖像。
        top_percent (float): 選擇暗通道中亮度最高的頂部百分比像素。

    Returns:
        numpy.ndarray: 大氣光 R, G, B 值。
    """
    # 找到暗通道中最亮的 top_percent 個像素點
    flat_dark = dark_channel.flatten()
    flat_img = img.reshape(img.shape[0] * img.shape[1], 3)
    
    # 找到最高值的索引
    search_size = int(flat_dark.shape[0] * top_percent)
    
    # 獲取索引並排序
    indices = np.argsort(flat_dark)[-search_size:]

    # 計算這些位置在原始圖像中的平均值，作為大氣光 A 的估算
    atmospheric_light = np.mean(flat_img[indices], axis=0)

    return atmospheric_light

def dark_channel_prior_dehazing(image_path, patch_size=15, omega=0.95, t0=0.1):
    """
    使用簡化的暗通道先驗 (DCP) 進行圖像去霧。

    Args:
        image_path (str): 圖像檔案的路徑。
        patch_size (int): 暗通道計算窗口大小。
        omega (float): 去霧深度參數 (通常 0.95 效果最好)。
        t0 (float): 透射率的最小值 (防止復原圖像過度曝光)。
    """
    try:
        # 讀取圖像
        img = cv2.imread(image_path)
        if img is None:
            print(f"錯誤：無法讀取圖像檔案 {image_path}")
            return
        
        # 轉換為浮點數 (為了計算方便)
        img_float = img.astype(np.float32) / 255.0

        # 1. 計算暗通道
        dark_channel = get_dark_channel((img_float * 255).astype(np.uint8), patch_size) / 255.0

        # 2. 估算大氣光 A
        A = get_atmospheric_light((img_float * 255).astype(np.uint8), (dark_channel * 255).astype(np.uint8)) / 255.0

        # 3. 估算透射率 t(x)
        # t(x) = 1 - ω * min_channel(I/A)
        # 這裡我們使用簡化版的透射率估算
        A_norm = np.maximum(A, 1e-6) # 防止除以零
        normalized_img = img_float / A_norm

        # 再次計算暗通道 (針對 I/A)
        t_numerator = get_dark_channel((normalized_img * 255).astype(np.uint8), patch_size) / 255.0

        # 計算透射率
        transmission = 1 - omega * t_numerator
        
        # 限制透射率的最小值 t0
        transmission = np.maximum(transmission, t0)
        
        # 4. 復原無霧圖像 J(x)
        # J(x) = (I(x) - A) / t(x) + A
        
        # 擴展 transmission 到三個通道
        t_3ch = cv2.merge([transmission, transmission, transmission])
        
        # 確保 t_3ch 不為零
        t_3ch = np.maximum(t_3ch, t0) 
        
        # 復原
        J = (img_float - A) / t_3ch + A
        
        # 限制範圍並轉換回 8 位元
        dehazed_img = np.uint8(np.clip(J * 255, 0, 255))
        
        # 顯示結果
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始霧霾圖像')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('DCP 去霧結果')
        axes[1].axis('off')

        plt.show()

        cv2.imwrite('dehazed_image.png', cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB))

    except Exception as e:
        print(f"處理圖像時發生錯誤: {e}")

# 範例使用 (請將 'your_hazy_image.jpg' 替換為您的空拍圖路徑)
dark_channel_prior_dehazing('./test1.tif')