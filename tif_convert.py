import PIL
import argparse
from PIL import Image

CUT_WIDTH = 640
CUT_HEIGHT = 640

def add_margin(pil_img, top, right, left, bottom, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def black_image_check(image):
    extrema = image.convert("L").getextrema()
    if extrema == (0, 0):
        return True
    return False

def cut_image(image):
    global CUT_WIDTH, CUT_HEIGHT
    img_width, img_height = image.size
    images = []
    for top in range(0, img_height, CUT_HEIGHT):
        for left in range(0, img_width, CUT_WIDTH):
            box = (left, top, min(left + CUT_WIDTH, img_width), min(top + CUT_HEIGHT, img_height))
            new_img = image.crop(box)
            # padding pixel not enough
            if new_img.size[0] < CUT_WIDTH or new_img.size[1] < CUT_HEIGHT:
                wd = max(CUT_WIDTH - new_img.size[0], 0)
                ht = max(CUT_HEIGHT - new_img.size[1], 0)
                new_img = add_margin(new_img, 0, wd, 0, ht, (0, 0, 0))
            if black_image_check(new_img):
                continue
            images.append(new_img)
    return images

def images_save(images, base_path: str, ext: str):
    for idx, img in enumerate(images):
        img.save(f"{base_path}_part{idx + 1}.{ext}")
        print(f"Saved {base_path}_part{idx + 1}.{ext}")

def convert_tif_to(folder_path, save_folder, file_type='png'):
    import os

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            path = os.path.join(save_folder, filename)
            tif_path = os.path.join(folder_path, filename)
            # png_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.png")
            
            with Image.open(tif_path) as img:
                imgs = cut_image(img)
                images_save(imgs, path, file_type)

                # print(f"Converted {tif_path} to {png_path}")

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Convert TIFF images to JPG and PNG formats.')
    parse.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing TIFF images.')
    parse.add_argument('--save_folder', type=str, help='Path to the folder to save converted images.')
    parse.add_argument('--type', type=str, required=True, choices=['jpg', 'png', 'both'], help='Type of conversion to perform.')
    args = parse.parse_args()

    folder_path = args.folder_path
    save_folder = args.save_folder if args.save_folder else folder_path
    if args.type == 'jpg':
        convert_tif_to(folder_path, save_folder, file_type='jpg')
    elif args.type == 'png':
        convert_tif_to(folder_path, save_folder, file_type='png')
    else:
        convert_tif_to(folder_path, save_folder, file_type='jpg')
        convert_tif_to(folder_path, save_folder, file_type='png') 