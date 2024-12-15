import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter

def convert_image_to_gray_matrix(image_path):
    """
    конвертация изображения в градации серого.
    :param image_path: путь к изображению, которое конвертируется.
    :return: Матрица (numpy array), представляющая изображение в градациях серого.
    """
    img = Image.open(image_path)
    gray_img = img.convert('L') # 'L' - режим для градаций серого
    gray_matrix = np.array(gray_img)
    return gray_matrix

def save_matrix_as_image(matrix, output_image_path):
    """
    сохранение изображения
    """
    new_image = Image.fromarray(matrix.astype(np.uint8))
    new_image.save(output_image_path)

def calculate_moments(histogram):
    """
    вычисление моменты и центральные моменты заданного гистограммы.
    :param histogram: представляющая распределение значений
    :return: кортеж из моментов (m1, m2, m3, m4) и центральных моментов (u1, u2, u3, u4)
    """
    m1 = np.sum(histogram * np.arange(len(histogram))) # Первый момент (среднее значение)
    print(f"Первый момент (M1): Σ (x_i * p_i) = Σ [{' ,'.join(map(str, histogram))}] * [{' ,'.join(map(str, np.arange(len(histogram))))}] = {m1}")
    m2 = np.sum(histogram * (np.arange(len(histogram)) ** 2))
    print(f"Второй момент (M2): Σ (x_i^2 * p_i) = Σ [{' ,'.join(map(str, histogram))}] * [{' ,'.join(map(str, np.arange(len(histogram)) ** 2))}] = {m2}")
    m3 = np.sum(histogram * (np.arange(len(histogram)) ** 3))
    print(f"Третий момент (M3): Σ (x_i^3 * p_i) = Σ [{' ,'.join(map(str, histogram))}] * [{' ,'.join(map(str, np.arange(len(histogram)) ** 3))}] = {m3}")
    m4 = np.sum(histogram * (np.arange(len(histogram)) ** 4))
    print(f"Четвертый момент (M4): Σ (x_i^4 * p_i) = Σ [{' ,'.join(map(str, histogram))}] * [{' ,'.join(map(str, np.arange(len(histogram)) ** 4))}] = {m4}")

    u1 = m1 / np.sum(histogram) # Первый центральный момент (среднее значение)
    print(f"Первый центральный момент (U1): M1 / N = {m1} / {np.sum(histogram)} = {u1}")
    u2 = (m2 / np.sum(histogram)) - (u1 ** 2) # Второй центральный момент (дисперсия)
    print(f"Второй центральный момент (U2): M2/N - U1^2 = ({m2} / {np.sum(histogram)}) - {u1 ** 2} = {m2 / np.sum(histogram)} - {u1 ** 2} = {u2}")
    u3 = (m3 / np.sum(histogram)) - (3 * u1 * u2) - (u1 ** 3) # Третий центральный момент
    print(f"Третий центральный момент (U3): M3/N - 3*U1*U2 - U1^3 = ({m3} / {np.sum(histogram)}) - ({3} * {u1} * {u2}) - {u1 ** 3} = {m3 / np.sum(histogram)} - {3 * u1 * u2} - {u1 ** 3} = {u3}")
    u4 = (m4 / np.sum(histogram)) - (4 * u1 * u3) - (6 * u1 ** 2 * u2) - (u2 ** 2) # Четвертый центральный момент
    print(f"Четвертый центральный момент (U4): M4/N - 4*U1*U3 - 6*U1^2*U2 - U2^2 = ({m4} / {np.sum(histogram)}) - ({4} * {u1} * {u3}) - ({6} * {u1} ** {2} * {u2}) - {u2 ** 2} = {m4 / np.sum(histogram)} - {4 * u1 * u3} - {6 * u1 ** 2 * u2} - {u2 ** 2} = {u4} ")

    return (m1, m2, m3, m4), (u1, u2, u3, u4)

def calculate_entropy(histogram):
    """
    вычисление энтропии
    :param histogram: представляющая распределение значений
    :return: энтропия (entropy)
    """
    total_pixels = np.sum(histogram) # сумма общее количество пикселей в гистограмме
    probabilities = histogram[histogram > 0] / total_pixels # вероятность для каждого ненулевого значения гистограммы
    entropy = -np.sum(probabilities * np.log2(probabilities))

    print("Расчет энтропии")
    print(f"Общее количество пикселей (N): Σ [{' ,'.join(map(str, histogram))}] = {total_pixels}")
    print(f"Вероятности (p_i): [{' ,'.join(map(str, histogram[histogram > 0]))}] / {total_pixels} =  [{' ,'.join(map(str, probabilities))}]")

    print("Формула для расчета энтропии: H = -Σ (p_i * log2(p_i))")
    results = []
    for p in probabilities:
        print(f"- ({p:.4f} * log2({p:.4f})) = {- (p * np.log2(p))}")
        results.append(- (p * np.log2(p)))

    print(f"H = {' + '.join(map(str, results))} = {entropy}")

    return entropy

def calculate_redundancy(entropy):
    """
    избыточность на основе энтропии
    :param entropy: энтропия
    :return: float: избыточность
    """
    max_entropy = np.log2(256) # max энтропия для 256 градаций серого (8 бит)
    redundancy = max_entropy - entropy # избыточность
    print(f"Формула для избыточности (R): R = H_max - H = {max_entropy} - {entropy} = {redundancy:.4f} бит")
    return redundancy

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

def truncated_block_coding(gray_matrix, block_size=8):
    """
    усеченное блочное кодирование к матрице
    :param: gray_matrix (numpy.ndarray): матрица градаций серого изображения, block_size (int): размер блока для кодирования
    :return: numpy.ndarray: Кодированная матрица градаций серого.
    """

    height, width = gray_matrix.shape # высота и ширина входной матрицы
    coded_matrix = np.zeros_like(gray_matrix)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = gray_matrix[i:i+block_size, j:j+block_size]
            coded_block = np.clip(block // 16 * 16, 0, 255)
            coded_matrix[i:i+block_size, j:j+block_size] = coded_block

    return coded_matrix

def calculate_standard_deviation(u2):
    return np.sqrt(u2)

# Геометрические преобразования
def translate(image, tx, ty):
    """Перенос изображения на (tx, ty)."""
    translated_image = Image.new("L", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if 0 <= x + tx < image.width and 0 <= y + ty < image.height:
                translated_image.putpixel((x + tx, y + ty), image.getpixel((x, y)))
    return translated_image

def rotate_and_scale(image, angle, scale):
    """Поворот и гомотетия изображения."""
    rotated_image = image.rotate(angle)
    width, height = rotated_image.size
    scaled_image = rotated_image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    return scaled_image

def scale_and_rotate(image, scale, angle):
    """Гомотетия и поворот изображения"""
    scaled_image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    rotated_image = scaled_image.rotate(angle)
    return rotated_image

def apply_filter(image, kernel):
    """свертка с заданным ядром"""
    kernel_size = int(np.sqrt(len(kernel)))  # размер ядра
    return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), np.array(kernel).flatten(), scale=None))

def median_filter(image, size):
    """медианный фильтр"""
    return image.filter(ImageFilter.MedianFilter(size=size))

def calculate_mse(original, filtered):
    """ среднеквадратичную ошибку (MSE) между оригинальным и отфильтрованным изображениями"""
    original_np = np.array(original)
    filtered_np = np.array(filtered)
    difference = original_np - filtered_np
    mse_value = np.mean(difference ** 2)
    # print(f"Среднеквадратичная ошибка (MSE): (1 / N) * Σ ( [{' ,'.join(map(str, original_np))}] - [{' ,'.join(map(str, filtered_np))}]) ^ 2")
    return mse_value

def calculate_psnr(original, filtered):
    """ пиковое отношение сигнал/шум (PSNR) между оригинальным и отфильтрованным изображениями"""
    mse = calculate_mse(original, filtered)
    # Если MSE равно 0, изображения идентичны, PSNR бесконечно велико
    if mse == 0:
        return float('inf')
    max_pixel = 255.0 # max значение пикселя для 8-битного изображения
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))

    # print(f"Пиковое отношение сигнал/шум (PSNR): 20 * log10({max_pixel} / {np.sqrt(mse)}) = 20 * {np.log10(max_pixel / np.sqrt(mse))} = {20 * np.log10(max_pixel / np.sqrt(mse))} дБ")

    return psnr_value

image_path = 'image.jpg'
gray_matrix = convert_image_to_gray_matrix(image_path)

# Построение гистограммы яркости
plot_brightness_histogram(gray_matrix)

# Вычисление моментов
histogram, _ = np.histogram(gray_matrix, bins=256, range=(0, 255))
(moments_m, moments_u) = calculate_moments(histogram)

# Вычисление энтропии и избыточности
entropy = calculate_entropy(histogram)
redundancy = calculate_redundancy(entropy)

coded_matrix = truncated_block_coding(gray_matrix) # Усеченное блочное кодирование

# Вычисление среднеквадратичного отклонения
std_dev = calculate_standard_deviation(moments_u[1])

# Вывод результатов
print("Вывод результатов")
print("Начальные моменты (m):", moments_m)
print("Центральные моменты (u):", moments_u)
print("Энтропия (Н):", entropy)
print("Избыточность (R):", redundancy)
print("Среднеквадратичное отклонение:", std_dev)

save_matrix_as_image(coded_matrix, 'coded_image.png')

# Применение геометрических преобразований
original_image = Image.open(image_path).convert('L')

# Перенос
translated_image = translate(original_image, tx=50, ty=30)
translated_image.save('translated_image.png')

# Поворот и гомотетия
rotated_scaled_image = rotate_and_scale(original_image, angle=45, scale=1.5)
rotated_scaled_image.save('rotated_scaled_image.png')

# Гомотетия и поворот
scaled_rotated_image = scale_and_rotate(original_image, scale=1.5, angle=45)
scaled_rotated_image.save('scaled_rotated_image.png')

# Применение фильтров
filtered_median_3x3 = median_filter(original_image, size=3)
filtered_median_5x5 = median_filter(original_image, size=5)
filtered_median_7x7 = median_filter(original_image, size=7)

# Сохранение отфильтрованных изображений
filtered_median_3x3.save("filtered_median_3x3.jpg")
filtered_median_5x5.save("filtered_median_5x5.jpg")
filtered_median_7x7.save("filtered_median_7x7.jpg")

# Сравнение
mse_3x3 = calculate_mse(original_image, filtered_median_3x3)
psnr_3x3 = calculate_psnr(original_image, filtered_median_3x3)

mse_5x5 = calculate_mse(original_image, filtered_median_5x5)
psnr_5x5 = calculate_psnr(original_image, filtered_median_5x5)

mse_7x7 = calculate_mse(original_image, filtered_median_7x7)
psnr_7x7 = calculate_psnr(original_image, filtered_median_7x7)

print(f"MSE 3x3 (Среднеквадратичная ошибка): {mse_3x3}, PSNR (пиковое отношение сигнал/шум): {psnr_3x3}")
print(f"MSE 5x5 (Среднеквадратичная ошибка): {mse_5x5}, PSNR (пиковое отношение сигнал/шум): {psnr_5x5}")
print(f"MSE 7x7 (Среднеквадратичная ошибка): {mse_7x7}, PSNR (пиковое отношение сигнал/шум): {psnr_7x7}")
