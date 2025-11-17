# import cv2
import os


path = "images/"

for idx, p in enumerate(os.listdir(path)):
    os.system("mv " + os.path.join(path, p) + " " + os.path.join(path, str(idx) + ".jpg"))
    print(idx)
    # img = cv2.imread(os
