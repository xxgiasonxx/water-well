import cv2

image = cv2.imread('./test.tif')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

equalize_img = cv2.equalizeHist(gray_image)

clahe = cv2.createCLAHE()
clahe_img = clahe.apply(gray_image)

def unsharp_masking(f, k = 1.0):
    g = f.copy()
    nr, nc = f.shape[:2]
    f_avg = cv2.GaussianBlur(f,(15, 15), 0)
    for x in range(nr):
        for y in range(nc):
            g_mask = int(f[x, y]) - int(f_avg[x, y])

    return g

clahe_img = unsharp_masking(clahe_img, k=10)

cv2.imshow('Sharpened Image', clahe_img)
cv2.imwrite('equalized_sharpened_image.jpg', clahe_img)