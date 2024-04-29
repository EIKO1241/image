import cv2
from skimage.metrics import structural_similarity as ssim

# 加载原始图像和重建图像
original_image = cv2.imread(r"output\woman.png")
reconstructed_image = cv2.imread(r"result\woman.png")
#统一图像尺寸
original_image = cv2.resize(original_image, (reconstructed_image.shape[1], reconstructed_image.shape[0]))
# 计算 PSNR
psnr = cv2.PSNR(original_image, reconstructed_image)
print("PSNR:", psnr)

# 计算 SSIM
ssim_value, _ = ssim(original_image, reconstructed_image, full=True, multichannel=True)
print("SSIM:", ssim_value)
