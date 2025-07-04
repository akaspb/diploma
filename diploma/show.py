import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import re


def read_pfm(file_name: str) -> np.ndarray:
    with open(file_name, 'rb') as file:
        file.readline()  # 'Pf' заголовок

        dims_line = file.readline().decode('utf-8').rstrip()
        dims_match = re.match(r'^(\d+)\s(\d+)$', dims_line)
        assert dims_match, 'can\'t read dimensions'

        width, height = map(int, dims_match.groups())

        scale_line = file.readline().decode('utf-8').rstrip()
        scale = float(scale_line)

        endian = '<' if scale < 0 else '>'
        data = np.fromfile(file, endian + 'f')
        data = data.reshape((height, width))
        return np.flipud(data)


def RMSElog(x, y, mask):
    x = x.copy()
    x[mask == 0] = 1
    y = y.copy()
    y[mask == 0] = 1
    return np.mean((np.log(x) - np.log(y))**2)**0.5


def show(input_folder: str, output_folder: str, image_idx: int):
    file_name = "{:0>8}".format(image_idx)

    ref_image = np.array(Image.open(f'{input_folder}images/{file_name}.jpg'))
    depth_gt = read_pfm(f'{input_folder}depth_gt/{file_name}.pfm')
    depth_est = read_pfm(f'{output_folder}depth_est/{file_name}.pfm')
    mask = np.array(Image.open(f'{input_folder}masks/{file_name}.png'), dtype=np.float32) > 10

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(ref_image)
    axs[0].set_title('Опорное изображение')
    axs[0].axis('off')

    axs[1].imshow(depth_gt)
    axs[1].set_title('Точная глубина')
    axs[1].axis('off')

    axs[2].imshow(depth_est)
    axs[2].set_title('Оценка глубины')
    axs[2].axis('off')

    depth_est[~mask] = 0

    axs[3].imshow(depth_est)
    axs[3].set_title('Оценка глубины c\nучётом маски')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()

    print('RMSE log:', RMSElog(depth_gt, depth_est, mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_folder", type=str, help="input data path")
    parser.add_argument("--output_folder", type=str, help="output path")
    parser.add_argument("--image_idx", type=int, help="reference image index")

    input_args = parser.parse_args()
    show(input_args.input_folder, input_args.output_folder, input_args.image_idx)
