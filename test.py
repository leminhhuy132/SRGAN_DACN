import cv2

image = cv2.imread('/home/minhhuy/Desktop/Python/SRGAN-PyTorch/data/Textdata/datasets/EN100/3.png', cv2.IMREAD_UNCHANGED)

height, width = image.shape[0:2]
print(height, width)
upscale_factor = 4
newsameH, newsameW = (height//upscale_factor)*upscale_factor, (width//upscale_factor)*upscale_factor
print(newsameH,newsameW)
image_original = cv2.resize(image, (newsameW, newsameH))
image_scale = cv2.resize(image_original, (newsameH//upscale_factor, newsameW//upscale_factor),
                   interpolation=cv2.INTER_CUBIC)

cv2.imshow('d', image_scale)
cv2.waitKey(0)
cv2.destroyWindow('d')