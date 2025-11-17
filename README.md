# 水井
## CV
- Dehazing 去霧
- DoG (Difference of Gaussians) 影像增強
- Lapiacian 拉普拉斯邊緣偵測
- Equalized 局部均衡化
## 轉檔
- tif_convert tif to jpg, png
- rename_image.py 重新命名
## 找水井

### annotate_wells_auto_fix
- --folder 
    - 型別:  str
    - 必須:  True
    - 說明:  包含 TIF 和 TFW 檔案的資料夾。
- --excel
    - 型別: str
    - 必須: True
    - 說明: "水井座標的 Excel 檔案路徑。
- --sheet
    - 型別: str
    - 預設: '工作表1'
    - 說明: "Excel 中的工作表名稱。
- --output
    - 型別: str
    - 必須: True
    - 說明: "輸出標註圖片的資料夾。
- --radius
    - 型別: int
    - 預設: 20
    - 說明: "標註圓圈的半徑 (像素)。
- --origin_img
    - 型別: bool
    - 預設: False
    - 說明: "是否輸出原始圖片（未標註）。
- --draw
    - 型別: bool
    - 預設: True
    - 說明: "是否在圖片上繪製標註。
- --cut_well
    - 型別: bool
    - 預設: True
    - 說明: "是否裁切水井區域。
- --cut_size
    - 型別: int
    - 預設: 640
    - 說明: "裁切水井區域的大小 (像素)。
