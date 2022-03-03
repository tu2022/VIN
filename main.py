import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys, os
from PIL import Image, ImageDraw

t2val = {}
ave1 = []
ave2 = []
number = 0


# 伽马变换
def transform(image, gamma, Li, Hi, Lo, Ho):
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            f = image.getpixel((x, y))
            p = ((f - Li) / (Hi - Li)) ** gamma
            p = p * (Ho - Lo) + Lo


# 二值化
def set_value(image, G):
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            g = image.getpixel((x, y))
            if g > G:
                t2val[(x, y)] = 1
            else:
                t2val[(x, y)] = 0


def threshold(image, number):
    if number == 0:
        number = np.median(image)
    ave1.clear()
    ave2.clear()
    for y in range(0, image.size[1]):
        for x in range(0, image.size[0]):
            g = image.getpixel((x, y))
            if g > number:
                ave1.append(g)
            else:
                ave2.append(g)
    ave(image, number)


def ave(image, number):
    number1 = np.mean(ave1)
    number2 = np.mean(ave2)
    if 2 * number > (number1 + number2) + 5:
        number = (number1 + number2) / 2
        threshold(image, number)
    elif 2 * number < (number1 + number2) - 5:
        number = (number1 + number2) / 2
        threshold(image, number)
    else:
        number = (number1 + number2) / 2
        set_value(image, number)


def saveImage(filename, size):
    image = Image.new("1", size)
    draw = ImageDraw.Draw(image)

    for x in range(0, size[0]):
        for y in range(0, size[1]):
            draw.point((x, y), t2val[(x, y)])

    image.save(filename)


def get_files(dir, suffix):
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            if suf in suffix:
                #res.append(filename)
                res.append(os.path.join(root, filename))
    return res


def uniform_image_size(list_path, width_size, height_size):
    image_list = get_files(list_path, ['.jpg'])
    total_len = len(image_list)
    print('total_label_len', total_len)
    for i in range(0, total_len):
        image_file = image_list[i]
        img = cv2.imread(image_file)
        cv2.imshow("input", img)
        cv2.waitKey(1)
        result = np.zeros(img.shape, dtype=np.float32)
        cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        print(result)
        cv2.imshow("norm", np.uint8(result*255.0))
        cv2.waitKey(1)
        os.remove(image_file)
        cv2.imwrite(image_file, np.uint8(result * 255.0))
    cv2.destroyAllWindows()

# filepath = ''
# for filename in os.listdir(filepath):
# image = Image.open('C:/Users/asus/Desktop/image1/%s' % filename).convert("L")
# threshold(image, 0)


def main():

    list_path = '/Users/tujiawen/Documents/VIN/fotos'
    width_size = 1280
    height_size = 720
    image_list = get_files(list_path, ['.jpg'])
    total_len = len(image_list)
    print('total_label_len', total_len)
    for i in range(0, total_len):
        image_file = image_list[i]
        img = cv2.imread(image_file)
        cv2.imshow("input", img)
        cv2.waitKey(0)
        #result = np.zeros(img.shape, dtype=np.float32)
        #cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #uniform_image_size(list_path, width_size, height_size)
        # 灰度化处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        cv2.waitKey(0)
        # 顶帽变换
        kernel = np.ones((5, 5), np.uint8)
        open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        img2 = cv2.subtract(gray, open)
        cv2.imshow('顶帽变换后', img2)
        cv2.waitKey(0)

        gamma = 3
        c = 255.0
        img_norm = img2/255.0
        img3 = c * np.power(img_norm, gamma)
        cv2.imshow("伽马变换", img3)
        cv2.waitKey(0)

        cv2.imwrite('processed %s' % image_file, img3)

        #ret, binary = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
        #cv2.imshow("二值化", binary)
        #cv2.waitKey(0)

    #filepath = '/Users/tujiawen/Documents/VIN/fotos'
    #for filename in os.listdir(filepath):
    # gray = Image.open('/Users/tujiawen/Documents/VIN/fotos/%s' % filename).convert('L')
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(gray)

    # Li, Hi, Lo, Ho = 0, 255, 0, 255
    # transform(gray, gamma, Li, Hi, Lo, Ho)
    # plt.figure()
    # plt.subplot(1, 3, 2)
    # plt.imshow(gray)

    # threshold(gray, 0)
    # plt.figure()
    # plt.subplot(1, 3, 3)
    # plt.imshow(gray)
    # saveImage('/Users/tujiawen/Documents/VIN/results/%s' % filename, gray.size)


if __name__ == '__main__':
    main()
