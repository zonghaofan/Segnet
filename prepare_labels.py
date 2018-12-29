import numpy as np
import os
import cv2
from PIL import Image
import math
import tifffile as tif

def del_no_need():
    need_path='./SegmentationClassPNG_OUT'

    del_path='./JPEGImages_OUT'

    need_names=os.listdir(need_path)
    del_names=os.listdir(del_path)

    for i in del_names:
        if i in need_names:
            pass
        else:
            del_path_name=os.path.join(del_path, i)
            os.remove(del_path_name)
def seg_image():
    path = './JPEGImages'
    outpath = './JPEGImages_OUT'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    images_names_list = [os.path.join(path, i) for i in os.listdir(path)]
    for image_name_list in images_names_list:
        img = cv2.imread(image_name_list)
        print(img.shape)
        cv2.imwrite(outpath+'/'+image_name_list.split('/')[-1].replace('jpg','png'),img)
def seg_label():
    path='./SegmentationClassPNG'
    outpath='./SegmentationClassPNG_OUT'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    images_names_list=[os.path.join(path,i) for i in os.listdir(path)]
    for num,image_name_list in enumerate(images_names_list):
        img = cv2.imread(image_name_list)
        # print(img.shape)
        # print(img)
        red = np.array([0, 0, 128])
        red_replace = np.array([0, 0, 0], dtype=np.int32)
        height, width, _ = img.shape

        yellow=np.array([0,128, 128])
        yellow_replace = np.array([1, 1,1], dtype=np.int32)

        green=np.array([0,128, 0])
        green_replace = np.array([2, 2, 2], dtype=np.int32)

        blue=np.array([128,0, 0])
        blue_replace = np.array([3, 3, 3], dtype=np.int32)

        back_gro = np.array([0, 0, 0])
        back_gro_replace = np.array([4, 4, 4], dtype=np.int32)
        # print(np.array([0, 0,0])==img[0,0,:])
        white_replace=np.array([255, 255, 255], dtype=np.int32)
        for i in range(height):
            for j in range(width):
                # print(a == img[i,j,:])
                if (red == img[i, j, :]).all():
                    img[i, j, :] = red_replace
                elif (yellow == img[i, j, :]).all():
                    img[i, j, :] = yellow_replace
                elif (green == img[i, j, :]).all():
                    img[i, j, :] = green_replace
                elif (blue == img[i, j, :]).all():
                     img[i, j, :] = blue_replace
                elif (back_gro == img[i, j, :]).all():
                     img[i, j, :] = back_gro_replace
        cv2.imwrite(outpath+'/'+image_name_list.split('/')[-1], img)
        print(image_name_list.split('/')[-1])
        print('{}img'.format(num))
def find_pic_index(img,array_list):
    img_sum = np.sum(img == array_list, axis=-1)
    y, x =np.where(img_sum == 3)
    index_list = np.stack((x, y), axis=-1)
    return index_list

def seg_label_new():
    path='./SegmentationClassPNG'
    outpath='./SegmentationClassPNG_OUT'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    images_names_list = [os.path.join(path, i) for i in os.listdir(path)]
    for num,image_name_list in enumerate(images_names_list):
        img = cv2.imread(image_name_list)
        # img_path='./SegmentationClassPNG/rec_1_0_20151031DandongRapideyeprjclip.png'
        # img=cv2.imread(img_path)
        print(img.shape)
        h,w,_=img.shape
        mask=np.zeros((h,w),dtype=np.int32)
        red = np.array([0, 0, 128])
        yellow = np.array([0, 128, 128])
        green = np.array([0, 128, 0])
        blue = np.array([128, 0, 0])
        back_gro = np.array([0, 0, 0])

        red_list=find_pic_index(img,red)
        yellow_list = find_pic_index(img, yellow)
        green_list = find_pic_index(img, green)
        blue_list = find_pic_index(img, blue)
        back_list = find_pic_index(img, back_gro)

        for i in red_list:
            mask[i[1],i[0]]=0
        for i in yellow_list:
            mask[i[1],i[0]]=1
        for i in green_list:
            mask[i[1],i[0]]=2
        for i in blue_list:
            mask[i[1],i[0]]=3
        for i in back_list:
            mask[i[1],i[0]]=4
        mask=np.expand_dims(mask,axis=-1)
        mask=np.concatenate((mask,mask,mask),axis=-1)
    # cv2.imwrite('./1.jpg', mask)
        cv2.imwrite(outpath+'/'+image_name_list.split('/')[-1], mask)
        print(image_name_list.split('/')[-1])
        print('{}img'.format(num))


def save_img(img):
    img = np.squeeze(img).astype(np.uint8)
    img = Image.fromarray(img)
    img.save('./2.png')
def test_visul_label():
    path='./testannot'
    # path='./rec_1_0_20170930RizhaoLSGQPlanet.png'
    output='./testannot_out'
    if not os.path.exists(output):
        os.mkdir(output)
    label_path=[os.path.join(path,i) for i in os.listdir(path)]
    for label_path_ in label_path:
        label = tif.imread(label_path_)
        # label=cv2.imread(label_path_)
        # label = cv2.resize(label, (512, 512),interpolation=cv2.INTER_NEAREST)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        print(label.shape)
        print(type(label))
        # label =label[:, :, 0]
        cmap = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [128, 128, 0],
        [0, 128, 0],
        [0, 0, 128]
        ]
        )
        y = label
        print(y)
        # print(y[100:120,:50])
        r = y.copy()
        print(r)
        g = y.copy()
        b = y.copy()
        # print('r=',r)
        for l in range(0, len(cmap)):
            r[y == l] = cmap[l, 0]
            g[y == l] = cmap[l, 1]
            b[y == l] = cmap[l, 2]
        print(r)

        # rgb = np.zeros((y.shape[0], y.shape[1], 3))
        # r=np.expand_dims(r,axis=-1)
        label=np.concatenate((np.expand_dims(b,axis=-1),np.expand_dims(g,axis=-1),
                              np.expand_dims(r,axis=-1)),axis=-1)
        print(label.shape)
        # rgb[:, :, 0] = r
        # rgb[:, :, 1] = g
        # rgb[:, :, 2] = b
        print(label)
        cv2.imwrite(output+'/'+label_path_.split('/')[-1],label)
    # save_img(label)

    #
    # if dir_path is None and filename is None:
    #     return (rgb * 255).astype(np.uint8)
    # else:
    #     return save_img(rgb, dir_path, filename)
if __name__ == '__main__':
    # seg_label()
    # seg_image()
    # del_no_need()
    # seg_label_new()
    test_visul_label()
    # path='./rec_1_0_20171022RizhaoLSGQPlanet.png'
    # img=cv2.imread(path)
    #
    # print(img[150:200,150:200])