import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/taylor_swift_noisy.png')


# Apply median filter
img_median = cv2.medianBlur(img, 5)
cv2.imwrite('images/taylor_swift_median.png', img_median)

plt.subplot(121)
plt.imshow(img_median)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title('Median Filter')

# Non-local means denoising
img_nlm = cv2.fastNlMeansDenoising(img, None, 10, 21, 41)
cv2.imwrite('images/taylor_swift_nlm.png', img_nlm)

plt.subplot(122)
plt.imshow(img_nlm)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title('NLM Denoising')
plt.show()
