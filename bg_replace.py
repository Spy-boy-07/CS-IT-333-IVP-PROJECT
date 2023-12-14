import cv2
import numpy as np


def binary_threshold(img, thresh):
    if len(img.shape) != 2:
        raise ValueError("Image should be gray scale")
    height, width = img.shape

    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i][j] = 255 if img[i][j] >= thresh else 0
    return result


def binary_erosion(img, kernel, iterations):
    if len(img.shape) != 2:
        raise ValueError("Image should be gray scale")

    if kernel.ndim != 2:
        raise ValueError("Kernel should be a 2d array")

    h, w = kernel.shape
    if h != w and w % 2 != 0:
        raise ValueError(
            "Kernel height and width should be same and size should be odd"
        )

    height, width = img.shape
    result = np.zeros((height, width), dtype=np.uint8)
    temp = np.copy(img)
    k = kernel * 255
    for i in range(iterations):
        for i in range(w, height - w):
            for j in range(w, width - w):
                w_half = w // 2
                flag = True

                for x in range(-1 * w_half, w_half + 1):
                    for y in range(-1 * w_half, w_half + 1):
                        if k[w_half + x][w_half + y] != temp[i + x][j + y]:
                            flag = False

                if flag:
                    result[i][j] = 255
        temp = np.copy(result)
    return result


def binary_dilation(img, kernel, iterations):
    if len(img.shape) != 2:
        raise ValueError("Image should be gray scale")

    if kernel.ndim != 2:
        raise ValueError("Kernel should be a 2d array")

    h, w = kernel.shape
    if h != w and w % 2 != 0:
        raise ValueError(
            "Kernel height and width should be same and size should be odd"
        )

    height, width = img.shape
    result = np.zeros((height, width), dtype=np.uint8)
    temp = np.copy(img)
    k = kernel * 255
    for i in range(iterations):
        for i in range(w, height - w):
            for j in range(w, width - w):
                w_half = w // 2
                flag = False

                for x in range(-1 * w_half, w_half + 1):
                    for y in range(-1 * w_half, w_half + 1):
                        if k[w_half + x][w_half + y] == temp[i + x][j + y]:
                            flag = True

                if flag:
                    result[i][j] = 255
        temp = np.copy(result)
    return result


def invert_mask(img):
    if len(img.shape) != 2:
        raise ValueError("Image should be gray scale")
    height, width = img.shape

    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i][j] = 255 if img[i][j] == 0 else 0
    return result


def replace_black_with_background_manual(foreground, background_path):
    background = cv2.imread(background_path)

    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

    mask = (gray_foreground > 1).astype(np.uint8) * 255

    mask_inv = 255 - mask

    # Replace black pixels in the foreground with corresponding background pixels manually using NumPy
    foreground_with_background = np.zeros_like(foreground)
    for c in range(3):
        foreground_with_background[:, :, c] = foreground[:, :, c] * (mask / 255)

    background_with_black_removed = np.zeros_like(background)
    for c in range(3):
        background_with_black_removed[:, :, c] = background[:, :, c] * (mask_inv / 255)

    # Combine the images
    result = foreground_with_background + background_with_black_removed

    # Display the result
    return result.astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread("img.jpeg")
    img = cv2.resize(img, (640, 640))
    cv2.imshow("Original Image", img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin = binary_threshold(gray, 180)
    cv2.imshow("Threshold Image", bin)

    kernel = np.ones((5, 5), np.uint8)
    opening = binary_erosion(bin, kernel, 1)
    closing = binary_dilation(opening, kernel, 1)
    cv2.imshow("Eroded Image", opening)
    cv2.imshow("Dilated Image", closing)

    mask = invert_mask(closing)
    cv2.imshow("Mask", mask)

    mask_3channel = np.stack([mask] * 3, axis=-1)
    foreground = np.bitwise_and(img, mask_3channel)
    cv2.imshow("Foreground Image", foreground)

    result = replace_black_with_background_manual(foreground, "bg.jpeg")
    cv2.imshow("Final Image", result)
    cv2.waitKey(0)
