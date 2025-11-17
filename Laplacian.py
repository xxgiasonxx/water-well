import cv2
import numpy as np
import matplotlib.pyplot as plt

def sharpen_with_laplacian(image_path):
    """
    使用拉普拉斯運算子對圖像進行銳化處理。

    Args:
        image_path (str): 圖像檔案的路徑。
    """
    try:
        # 讀取圖像
        img = cv2.imread(image_path)
        if img is None:
            print(f"錯誤：無法讀取圖像檔案 {image_path}")
            return

        # 1. 轉換為灰度圖 (拉普拉斯運算子通常在灰度圖上操作)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. 應用拉普拉斯運算子
        # cv2.CV_64F 確保運算過程中保留浮點數精度，防止溢位
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # 3. 轉換回 8 位元整數並取絕對值 (可選，用於視覺化 Laplacian 結果)
        laplacian_8u = np.uint8(np.absolute(laplacian))

        # 4. 進行銳化 (Unsharp Masking 的概念：原始圖 + 邊緣)
        # 這裡我們將原始彩色圖和 Laplacian 提取的邊緣結合，但通常需要更複雜的加權處理
        # 簡化方法：將 Laplacian 邊緣疊加到原始圖像上，但 Laplacian 輸出是灰度，
        # 實際銳化通常是將原始圖像減去高斯模糊或直接用 Laplacian 濾波器核心處理。

        # 更實用的銳化方法：
        # 將原始圖像轉換為浮點數
        img_float = img.astype(np.float32)

        # 應用拉普拉斯濾波器核心 (Kernel) 進行處理 (這是更標準的銳化方法)
        # 定義一個拉普拉斯濾波器核心
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]]) / 9.0  # 正規化可避免過度銳化

        # 應用 2D 卷積
        sharp_img_float = cv2.filter2D(img_float, -1, kernel)

        # 銳化後的圖像 = 原始圖像 + 濾波結果 (將邊緣加回去)
        sharp_img_float = img_float + sharp_img_float * 1.5 # 0.5 是銳化強度

        # 裁切數值到 [0, 255] 範圍內
        sharp_img_float = np.clip(sharp_img_float, 0, 255)

        # 轉換回 8 位元整數
        sharp_img = np.uint8(sharp_img_float)


        # 顯示結果
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始圖像')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title('拉普拉斯銳化結果')
        axes[1].axis('off')

        plt.show()

        cv2.imwrite('sharpened_image.jpg', sharp_img)

    except Exception as e:
        print(f"處理圖像時發生錯誤: {e}")

# 範例使用 (請將 'your_blurred_image.jpg' 替換為您的空拍圖路徑)
sharpen_with_laplacian('./test.tif')