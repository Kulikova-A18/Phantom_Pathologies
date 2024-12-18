import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter

def load_image_as_gray(image_path):
 img = Image.open(image_path)
 grayscale_image = img.convert('L')  # 'L' - режим для градаций серого
 return np.array(grayscale_image)

def save_image_from_matrix(matrix, output_path):
 img = Image.fromarray(matrix.astype(np.uint8))
 img.save(output_path)

def compute_image_moments(histogram):
    with open('compute_image_moments.log', 'w', encoding='utf-8') as log_file:
        total_pixels = np.sum(histogram)
        log_file.write(f"Общее количество пикселей: {total_pixels}\n\n")
        first_moment = np.sum(histogram * np.arange(len(histogram)))
        log_file.write(f"Первый момент (M1): M1 = Σ (h[i] * i) = {first_moment}\n")
        second_moment = np.sum(histogram * (np.arange(len(histogram)) ** 2))
        log_file.write(f"Второй момент (M2): M2 = Σ (h[i] * i^2) = {second_moment}\n")
        third_moment = np.sum(histogram * (np.arange(len(histogram)) ** 3))
        log_file.write(f"Третий момент (M3): M3 = Σ (h[i] * i^3) = {third_moment}\n")
        fourth_moment = np.sum(histogram * (np.arange(len(histogram)) ** 4))
        log_file.write(f"Четвертый момент (M4): M4 = Σ (h[i] * i^4) = {fourth_moment}\n\n")
        central_first = first_moment / total_pixels
        log_file.write(f"Центральный первый момент: μ1 = M1 / N = {first_moment} / {total_pixels} = {central_first}\n")
        central_second = (second_moment / total_pixels) - (central_first ** 2)
        log_file.write(f"Центральный второй момент: μ2 = (M2 / N) - μ1^2 = ({second_moment} / {total_pixels}) - ({central_first}^2) = {central_second}\n")
        central_third = (third_moment / total_pixels) - (3 * central_first * central_second) - (central_first ** 3)
        log_file.write(f"Центральный третий момент: μ3 = (M3 / N) - 3 * μ1 * μ2 - μ1^3 = ({third_moment} / {total_pixels}) - 3 * {central_first} * {central_second} - ({central_first}^3) = {central_third}\n")
        central_fourth = (fourth_moment / total_pixels) - (4 * central_first * central_third) - (6 * central_first ** 2 * central_second) - (central_second ** 2)
        log_file.write(f"Центральный четвертый момент: μ4 = (M4 / N) - 4 * μ1 * μ3 - 6 * μ1^2 * μ2 - μ2^2 = ({fourth_moment} / {total_pixels}) - 4 * {central_first} * {central_third} - 6 * ({central_first}^2) * {central_second} - ({central_second}^2) = {central_fourth}\n")
    return (first_moment, second_moment, third_moment, fourth_moment), (central_first, central_second, central_third, central_fourth)

def calculate_image_entropy(histogram):
 total_pixels = np.sum(histogram)
 probabilities = histogram[histogram > 0] / total_pixels
 entropy = -np.sum(probabilities * np.log2(probabilities))
 return entropy

def calculate_redundancy_from_entropy(entropy):
 max_entropy = np.log2(256)
 return max_entropy - entropy

def block_based_quantization(matrix, block_size=8):
 height, width = matrix.shape
 quantized_matrix = np.zeros_like(matrix)
 for i in range(0, height, block_size):
  for j in range(0, width, block_size):
   block = matrix[i:i+block_size, j:j+block_size]
   quantized_block = np.clip(block // 16 * 16, 0, 255)
   quantized_matrix[i:i+block_size, j:j+block_size] = quantized_block
   return quantized_matrix

def compute_standard_deviation(variance):
 return np.sqrt(variance)

def shift_image(image, tx, ty):
    translated_image = Image.new("L", image.size)
    with open('shift_image.log', 'w') as log_file:
        log_file.write(f"Начинаем с изображения размером: {image.size}\n")
        log_file.write(f"Смещение по X: {tx}, Смещение по Y: {ty}\n")
        for x in range(image.width):
            for y in range(image.height):
                new_x = x + tx
                new_y = y + ty
                if 0 <= new_x < image.width and 0 <= new_y < image.height:
                    pixel_value = image.getpixel((x, y))
                    translated_image.putpixel((new_x, new_y), pixel_value)
                    log_file.write(f"Перемещаем пиксель из ({x}, {y}) в ({new_x}, {new_y}) с значением: {pixel_value}\n")
                else:
                    log_file.write(f"Пиксель ({x}, {y}) не может быть перемещен в ({new_x}, {new_y}), выходит за пределы изображения.\n")
    return translated_image

def rotate_and_resize(image, angle, scale):
    with open('rotate_and_resize.log', 'w') as log_file:
        log_file.write(f"Исходный размер изображения: {image.size}\n")
        log_file.write(f"Угол поворота: {angle} градусов\n")
        log_file.write(f"Коэффициент масштабирования: {scale}\n")

        rotated_image = image.rotate(angle)
        width, height = rotated_image.size
        log_file.write(f"Размер изображения после поворота: {rotated_image.size}\n")

        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = rotated_image.resize((new_width, new_height), Image.LANCZOS)
        log_file.write(f"Размер изображения после изменения размера: {resized_image.size}\n")
    return resized_image


def apply_kernel_filter(image, kernel):
 kernel_size = int(np.sqrt(len(kernel)))
 return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), np.array(kernel).flatten(), scale=None))

def median_filter(image, size):
 return image.filter(ImageFilter.MedianFilter(size=size))

def calculate_mse_between_images(original, filtered):
    with open('calculate_mse_between_images.log', 'w') as log_file:
        original_array = np.array(original)
        filtered_array = np.array(filtered)
        log_file.write(f"Размер оригинального изображения: {original_array.shape}\n")
        log_file.write(f"Размер отфильтрованного изображения: {filtered_array.shape}\n")
        if original_array.shape != filtered_array.shape:
            log_file.write("Ошибка: размеры изображений не совпадают!\n")
            return None
        difference = original_array - filtered_array
        log_file.write(f"Разница между изображениями:\n{difference}\n")
        mse = np.mean(difference ** 2)
        log_file.write(f"Среднеквадратичная ошибка (MSE): {mse}\n")
    return mse

def calculate_psnr_between_images(original, filtered):
    with open('calculate_psnr_between_images.log', 'w') as log_file:
        mse = calculate_mse_between_images(original, filtered)
        log_file.write(f"Среднеквадратичная ошибка (MSE): {mse}\n")
        if mse == 0:
            log_file.write("MSE равно 0, PSNR бесконечен.\n")
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        log_file.write(f"Максимальное значение пикселя: {max_pixel}\n")
        log_file.write(f"Вычисленный PSNR: {psnr} дБ\n")
    return psnr

def plot_brightness_histogram(gray_matrix):
    histogram, bin_edges = np.histogram(gray_matrix, bins=256, range=(0, 255))

    plt.figure(figsize=(10, 5))
    plt.title("Гистограмма яркости")
    plt.xlabel("Яркость (0-255)")
    plt.ylabel("Количество пикселей")
    plt.xlim([0, 255])
    plt.plot(bin_edges[0:-1], histogram, color='red')

    max_brightness = np.argmax(histogram)
    max_count = histogram[max_brightness]

    plt.annotate(f'Максимум: {max_brightness}\nКоличество: {max_count}',
                 xy=(max_brightness, max_count),
                 xytext=(max_brightness + 10, max_count - 50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    x_ticks = np.concatenate((np.arange(0, 100, 10), np.arange(100, 256, 20)))
    plt.xticks(x_ticks)

    for i in range(256):
        plt.gca().add_patch(patches.Rectangle((i, 0), 1, max(histogram), color=str(i/255), alpha=0.8))

    plt.grid()
    plt.show()

image_path = 'image.jpg'
gray_image_matrix = load_image_as_gray(image_path)

plot_brightness_histogram(gray_image_matrix)

histogram, _ = np.histogram(gray_image_matrix, bins=256, range=(0, 255))
(moments, central_moments) = compute_image_moments(histogram)

entropy_value = calculate_image_entropy(histogram)
redundancy_value = calculate_redundancy_from_entropy(entropy_value)
quantized_image = block_based_quantization(gray_image_matrix)
std_dev_value = compute_standard_deviation(central_moments[1])

print("Моменты (m):", moments)
print("Центральные моменты (u):", central_moments)
print("Энтропия:", entropy_value)
print("Избыточность:", redundancy_value)
print("Стандартное отклонение:", std_dev_value)
save_image_from_matrix(quantized_image, 'quantized_image.png')

image_for_transformations = Image.open(image_path).convert('L')
shifted_image = shift_image(image_for_transformations, tx=50, ty=30)
shifted_image.save('shifted_image.png')
rotated_resized_image = rotate_and_resize(image_for_transformations, angle=-90, scale=1.5)
rotated_resized_image.save('rotated_resized_image.png')
filtered_image_3x3 = median_filter(image_for_transformations, size=3)
filtered_image_3x3.save("filtered_3x3.jpg")
mse_3x3 = calculate_mse_between_images(image_for_transformations, filtered_image_3x3)
psnr_3x3 = calculate_psnr_between_images(image_for_transformations, filtered_image_3x3)

# Сглаживание
smoothed = cv.blur(gray_image_matrix, (3, 3))
# Выделение линий
line_kernel = np.array([[-1, -1, -1],
                         [ 2,  2,  2],
                         [-1, -1, -1]], dtype=np.float32)
lines = cv.filter2D(gray_image_matrix, -1, line_kernel)
# ВЧ фильтр
sharp_kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)
hfq = cv.filter2D(gray_image_matrix, -1, sharp_kernel)
# Фильтр Лапласа
laplace = cv.Laplacian(gray_image_matrix, cv.CV_64F, ksize=3)
laplace = cv.convertScaleAbs(laplace)
# Медианный фильтр
median = cv.medianBlur(gray_image_matrix, 3)
# Перепад (Собель)
gx = cv.Sobel(gray_image_matrix, cv.CV_64F, 1, 0, ksize=3)
gy = cv.Sobel(gray_image_matrix, cv.CV_64F, 0, 1, ksize=3)
grad = cv.magnitude(gx, gy)
grad = cv.convertScaleAbs(grad)

cv.imshow('Original image (bw) and Smoothing filter', np.hstack((gray_image_matrix, smoothed)))
cv.imshow('Original image (bw) and Line extraction', np.hstack((gray_image_matrix, lines)))
cv.imshow('Original image (bw) and High-pass filter', np.hstack((gray_image_matrix, hfq)))
cv.imshow('Original image (bw) and Laplace filter', np.hstack((gray_image_matrix, laplace)))
cv.imshow('Original image (bw) and Median filter', np.hstack((gray_image_matrix, median)))
cv.imshow('Original image (bw) and Drop (Sobel)', np.hstack((gray_image_matrix, grad)))

cv.waitKey(0)
cv.destroyAllWindows()
