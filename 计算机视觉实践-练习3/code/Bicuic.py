from PIL import Image

# 打开原始图像
image = Image.open(r"Set5\woman.png")

# 指定目标尺寸
target_width = int(image.width * 0.5)  # 50% 的宽度
target_height = int(image.height * 0.5)  # 50% 的高度

# 进行 Bicubic 下采样
resized_image = image.resize((target_width, target_height), Image.BICUBIC)

# 保存处理后的图像
resized_image.save(r"output\woman.png")
