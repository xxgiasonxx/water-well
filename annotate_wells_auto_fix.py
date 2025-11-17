# annotate_wells_auto_fix.py
import os
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

# rasterio optional (if not available, script still works using tfw)
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

def parse_tfw_raw(tfw_path):
    """讀原始 tfw，不做任何假設轉換（直接回傳 A,D,B,E,C,F）"""
    with open(tfw_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    if len(lines) < 6:
        raise ValueError("Invalid TFW file - fewer than 6 lines")
    A = float(lines[0])
    D = float(lines[1])
    B = float(lines[2])
    E = float(lines[3])
    C = float(lines[4])
    F = float(lines[5])
    return A, D, B, E, C, F

def tfw_center_to_corner(tfw):
    """
    【修正時已不再使用此函數，但保留定義】
    把 tfw 假設為 pixel-center (如 Pix4D) 的情況，轉成 pixel-corner (GDAL style)
    """
    A, D, B, E, C, F = tfw
    C_new = C - 0.5 * A - 0.5 * B
    F_new = F - 0.5 * D - 0.5 * E
    return (A, D, B, E, C_new, F_new)

def geo_to_pixel_affine(X, Y, tfw):
    """一般仿射反算 (i,j) 浮點 (col,row) — 未做 int rounding"""
    A, D, B, E, C, F = tfw
    det = A * E - B * D
    if abs(det) < 1e-12:
        raise ValueError("Singular transform (determinant ~ 0)")
    i = (E * (X - C) - B * (Y - F)) / det
    j = (A * (Y - F) - D * (X - C)) / det
    return float(i), float(j)

def compute_bbox_from_tfw(tfw, width, height):
    """回傳 (minX, maxX, minY, maxY) — 使用 pixel centers範圍 """
    # 這裡的邏輯是假設 tfw 是 Corner-based 的 GDAL/Rasterio 風格變換
    A, D, B, E, C, F = tfw
    
    # 為了簡潔，我們只考慮非旋轉情況來計算中心範圍 (這是原始程式碼的簡化邏輯)
    if abs(B) < 1e-12 and abs(D) < 1e-12:
        # Corner-based (C,F) is top-left corner. Center is at C + 0.5*A, F + 0.5*E
        minX = C + 0.5 * A
        maxX = C + (width - 0.5) * A
        maxY = F + 0.5 * E  # E is usually negative
        minY = F + (height - 0.5) * E
    else:
        # 旋轉情況下，邊界計算會更複雜，保留原始的保守估計
        minX = C + 0.5 * A + 0.5 * B
        maxX = C + (width - 0.5) * A + (height - 0.5) * B
        maxY = F + 0.5 * D + 0.5 * E
        minY = F + (width - 0.5) * D + (height - 0.5) * E
        
    return min(minX, maxX), max(minX, maxX), min(minY, maxY), max(minY, maxY)

def try_read_tif_transform(tif_path):
    """若有 rasterio，讀取 tiff 的 transform (Affine) 回傳 (A,B,C,D,E,F) 同 tfw 格式 (A,D,B,E,C,F)"""
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(tif_path) as src:
            t = src.transform  # Affine(a, b, c, d, e, f)
            # Convert to tfw-like tuple: A = a, D = d, B = b, E = e, C = c, F = f
            # 此 Affine 變換是 GDAL 標準的 Corner-based
            return (t.a, t.d, t.b, t.e, t.c, t.f)
    except Exception:
        return None

# =======================================================================
# 核心修正區域：簡化變換選擇邏輯
# =======================================================================
def auto_fix_and_annotate(tif_path, tfw_path, wells, out_path, mark_radius=20):
    """自動修復潛在的 TFW 偏移並標註，並在像素轉換後應用 0.5 像素平移。"""
    
    tfw_raw = parse_tfw_raw(tfw_path)
    internal = try_read_tif_transform(tif_path)
    
    use_transform = None
    
    # 邏輯保持修正後的簡潔：優先使用 GeoTIFF 內部變換，其次是原始 TFW。
    if internal is not None:
        use_transform = internal 
    else:
        use_transform = tfw_raw
    
    # open image and draw
    img = Image.open(tif_path).convert("RGBA")
    width, height = img.size
    draw = ImageDraw.Draw(img)

    found = []
    
    # 假設 use_transform 是一個 Corner-based 變換 (GDAL 標準)
    for well_id, X, Y in wells:
        # 1. 計算 Corner-based 的浮點像素坐標 (i, j)
        colf_corner, rowf_corner = geo_to_pixel_affine(X, Y, use_transform)
        
        # 2. 【關鍵修正】強制應用 0.5 像素平移 (Corner-to-Center)
        # 由於您報告標記點在物件左上角，表示計算出的 (i, j) 偏小，
        # 應平移 +0.5 來到達像素中心。
        colf = colf_corner + 0.5
        rowf = rowf_corner + 0.5
        
        # 3. 四捨五入到最近的整數像素索引
        coli = int(round(colf))
        rowi = int(round(rowf))
        
        # check inside image
        if coli < 0 or coli >= width or rowi < 0 or rowi >= height:
            continue
        print(args.draw)
        if args.draw:
            draw.ellipse((coli-mark_radius, rowi-mark_radius, coli+mark_radius, rowi+mark_radius), outline='red', width=3)
        if args.cut_well:
            well_crop = cut_well_region(img, coli, rowi, size=args.cut_size)
            if black_image_check(well_crop):
                continue
            crop_path = out_path.replace('.png', f'_well_{well_id}.png')
            well_crop.save(crop_path)
        found.append((well_id, coli, rowi, colf, rowf))
            
    # save annotated if any
    if found and args.origin_img:
        img.save(out_path)
        
    return found, (tfw_raw, None, internal, use_transform)

def cut_well_region(img, center_x, center_y, size=640):
    """從 img 裁切以 (center_x, center_y) 為中心，size x size 大小的區域"""
    left = max(center_x - size // 2, 0)
    upper = max(center_y - size // 2, 0)
    right = min(center_x + size // 2, img.width)
    lower = min(center_y + size // 2, img.height)
    img = img.crop((left, upper, right, lower))
    if img.width < size or img.height < size:
        new_img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        new_img.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return new_img
    return img

def black_image_check(img):
    """檢查圖片是否超過97%透明 (透明)"""
    alpha = img.split()[-1]
    alpha_data = np.array(alpha)
    transparent_pixels = np.sum(alpha_data == 0)
    total_pixels = alpha_data.size
    if transparent_pixels / total_pixels >= 0.97:
        return True
    return False

def read_wells_from_excel(excel_file, sheet):
    df = pd.read_excel(excel_file, sheet_name=sheet)
    wells = []
    for idx, row in df.iterrows():
        # 確保 Excel 中的欄位名稱與此處匹配
        wells.append((row['序號'], float(row['TWD97E']), float(row['TWD97N'])))
    return wells

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help="包含 TIF 和 TFW 檔案的資料夾。")
    parser.add_argument('--excel', type=str, required=True, help="水井座標的 Excel 檔案路徑。")
    parser.add_argument('--sheet', type=str, default='工作表1', help="Excel 中的工作表名稱。")
    parser.add_argument('--output', type=str, required=True, help="輸出標註圖片的資料夾。")
    parser.add_argument('--radius', type=int, default=20, help="標註圓圈的半徑 (像素)。")
    parser.add_argument('--origin_img', type=bool, default=False, help="是否輸出原始圖片（未標註）。")
    parser.add_argument('--draw', type=bool, default=False, help="是否在圖片上繪製標註。")
    parser.add_argument('--cut_well', type=bool, default=True, help="是否裁切水井區域。")
    parser.add_argument('--cut_size', type=int, default=640, help="裁切水井區域的大小 (像素)。")
    args = parser.parse_args()


    print(args.draw)

    os.makedirs(args.output, exist_ok=True)
    wells = read_wells_from_excel(args.excel, args.sheet)
    tifs = [f for f in os.listdir(args.folder) if f.lower().endswith('.tif')]
    
    print(f"找到 {len(tifs)} 個 TIF 檔案，共 {len(wells)} 個水井座標。")
    
    for tif in tqdm(tifs):
        tif_path = os.path.join(args.folder, tif)
        # 假設 TFW 檔案名與 TIF 檔案名匹配
        tfw_path = tif_path.replace('.tif', '.tfw').replace('.TIF', '.TFW') 
        
        if not os.path.exists(tfw_path):
            # print(f"Warning: TFW file not found for {tif}. Skipping.")
            continue
            
        # 輸出檔案使用 PNG 格式以保留透明度，並避免與原始 TIF 衝突
        out_path = os.path.join(args.output, tif.replace('.tif','_annotated.png').replace('.TIF','_annotated.png'))
        
        # 呼叫修正後的函數
        found, debug = auto_fix_and_annotate(tif_path, tfw_path, wells, out_path, args.radius)
        
        # Debug 輸出 (可選)
        # print(f"\n--- Debug {tif} ---")
        # print(f"Raw TFW: {debug[0]}")
        # print(f"Internal (Rasterio): {debug[2]}")
        # print(f"Used Transform: {debug[3]}")
        # print(f"-------------------")
        
        if found:
            print(f"Processed {tif}: found and annotated {len(found)} wells (Saved to {os.path.basename(out_path)})")
        else:
            print(f"Processed {tif}: found 0 wells (Output image not saved)")