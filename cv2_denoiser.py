import cv2
import matplotlib.pyplot as plt
import time
import multiprocessing

# Set number of CPUs
cv2.setNumThreads(int(multiprocessing.cpu_count() * 0.75))

img = cv2.imread('images/taylor_swift_noisy.png')


# Apply median filter
median_filter_stime = time.time()
img_median = cv2.medianBlur(img, 5)
median_filter_time = time.time() - median_filter_stime
cv2.imwrite('images/taylor_swift_median.png', img_median)
print('Median filter time: ', round(median_filter_time, 6), " seconds")

plt.subplot(121)
plt.imshow(img_median)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title('Median Filter')

# Non-local means denoising
nlm_stime = time.time()
img_nlm = cv2.fastNlMeansDenoising(img, None, 10, 21, 41)
nlm_time = time.time() - nlm_stime
cv2.imwrite('images/taylor_swift_nlm.png', img_nlm)
print('NLM time: ', round(nlm_time, 6), " seconds")

plt.subplot(122)
plt.imshow(img_nlm)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.title('NLM Denoising')
plt.show()

