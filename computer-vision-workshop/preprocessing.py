import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

#görüntünün okunması
image_path="images/sample.jpg"
image=cv2.imread(image_path)
if image is None: print("image okunamadı")

#görüntünün görüntülenmesi
cv2.imshow("original image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# görüntü boyutu
height, width, channels =image.shape
print(f"height: {height}, width: {width}, channels: {channels}")

# görüntünün yeniden boyutlandırılması
resized_image =cv2.resize(image, (640,360))
print(f"resized_image.shape: {resized_image.shape}")
cv2.waitKey(0)
cv2.destroyAllWindows()

# crop
resized_height, resized_width = resized_image.shape[:2]
x1 = int(resized_width * 0.2)
x2 = int(resized_width * 0.8)
y1 = int(resized_height * 0.2)
y2 = int(resized_height * 0.8)
cropped_image = resized_image[y1:y2, x1:x2]
cv2.imshow("cropped_image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#bgr to rgb
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_image)
plt.show()

#bgr to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image)
plt.show()

#normalizasyon: 0-255 -> 0-1
normalized_image = resized_image.astype(np.float32) /255.0
print(normalized_image.min())
print(normalized_image.max())

#blur-filtreleme
blurred_image = cv2.GaussianBlur(resized_image, (15,15), 0)
cv2.imshow("blurred_image", blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#işlenmiş görüntülerin kaydedilmesi
cv2.imwrite("outputs/resized.jpg", cropped_image)