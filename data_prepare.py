#coding:utf-8
from tensorflow.python.framework import ops
import tensorflow as tf
import os
import numpy as np
FLAGS = tf.flags.FLAGS


#####################data prepare#############################
def get_file_names(file_dir,name):
    imgs=os.listdir(file_dir+'/'+name)
    labels=os.listdir(file_dir+'/'+name+'annot')
    img_names=[file_dir+'/'+name+'/'+x  for x in imgs]
    label_names=[file_dir+'/'+name+'annot'+'/'+x for x in labels]
    return sorted(img_names),sorted(label_names)

def data_augmentation(image,label,aug=False):
    if aug:
        # label = tf.image.resize_images(label, (img_size, img_size))
        coefficients=np.random.uniform(0.6,1.4)
        image = tf.image.random_brightness(image, coefficients)
        image = tf.image.random_hue(image, max_delta=0.05)
        # 设置随机的对比度
        image=tf.image.random_contrast(image,lower=0.99,upper=1.0)

        image_label = tf.concat([image,label],axis = -1)
        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

        image = maybe_flipped[:, :, :-1]
        label = maybe_flipped[:, :, -1:]
        # label=tf.image.resize_images(label, (label_size, label_size))
    else:
        pass
    return image, label
##########################################################
def get_data_label_batch(imgs_dir,labels_dir,augmen):
    imgs_tensor=ops.convert_to_tensor(imgs_dir,dtype=tf.string)

    labels_tensor=ops.convert_to_tensor(labels_dir,dtype=tf.string)
    filename_queue=tf.train.slice_input_producer([imgs_tensor,labels_tensor])
    
    image_filename = filename_queue[0]
    label_filename = filename_queue[1]
    
    imgs_values=tf.read_file(image_filename)
    label_values=tf.read_file(label_filename)


    imgs_decorded=tf.image.decode_png(imgs_values,channels=3)#return RGB
    labels_decorded=tf.image.decode_png(label_values,channels=1)

    # imgs_reshaped=tf.reshape(imgs_decorded,[FLAGS.img_height,FLAGS.img_width,3])
    # labels_reshaped=tf.reshape(labels_decorded,[FLAGS.img_height,FLAGS.img_width,1])
    imgs_reshaped = tf.image.resize_images(imgs_decorded, (FLAGS.height, FLAGS.width),)
    labels_reshaped = tf.image.resize_images(labels_decorded, (FLAGS.height, FLAGS.width),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    imgs_reshaped.set_shape([FLAGS.height, FLAGS.width, 3])
    labels_reshaped.set_shape([FLAGS.height, FLAGS.width, 1])
    imgs_reshaped = tf.cast(imgs_reshaped, tf.float32)
    labels_reshaped = tf.cast(labels_reshaped, tf.float32)

    img,label=data_augmentation(imgs_reshaped,labels_reshaped,aug=augmen)
    img = tf.cast(img, tf.float32)
    label = tf.cast(label, tf.int32)
    images_batch, labels_batch = tf.train.shuffle_batch([img,label],
                                                           batch_size=FLAGS.batch_size,
                                                           # num_threads=6,\
                                                           capacity=5 * FLAGS.batch_size,
                                                           min_after_dequeue=2 * FLAGS.batch_size,
                                                           allow_smaller_final_batch=True)
    return images_batch, labels_batch 