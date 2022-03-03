import cv2
import numpy as np
# from matplotlib import pyplot as plt
import sys, os
from PIL import Image, ImageDraw

t2val = {}
ave1 = []
ave2 = []
number = 0


# 伽马变换
def transform(image, gamma, Li, Hi, Lo, Ho):
    for y in range(0, image.shape[1]):
        for x in range(0, image.shape[0]):
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


#def saveImage(filename, size):
    #image = Image.new("1", size)
    #draw = ImageDraw.Draw(image)

    #for x in range(0, size[0]):
        #for y in range(0, size[1]):
            #draw.point((x, y), t2val[(x, y)])

    #image.save(filename)


# filepath = ''
# for filename in os.listdir(filepath):
# image = Image.open('C:/Users/asus/Desktop/image1/%s' % filename).convert("L")
# threshold(image, 0)
# saveImage('C:/Users/asus/Desktop/imageresult/%s' % filename, image.size)

def main():
    # 灰度化处理
    #gray = Image.open('1.jpg').convert('L')
    img = cv2.imread('1.jpg', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    # 顶帽变换
    kernel = np.ones((5, 5), np.uint8)
    open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # dst = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # img2 = gray-dst
    img2 = cv2.subtract(gray, open)
    cv2.imshow('顶帽变换后', img2)
    cv2.waitKey(0)
    #print(img.size)

    gamma = 3
    c = 255.0
    #Li, Hi, Lo, Ho = 0, 255, 0, 255
    #transform(img2, gamma, Li, Hi, Lo, Ho)
    img_norm = img2/255.0
    img2 = c * np.power(img_norm, gamma)
    cv2.imshow("伽马变换", img2)
    cv2.waitKey(0)

    img3 = Image.open(img2)
    threshold(img3, 0)


if __name__ == '__main__':
    main()
