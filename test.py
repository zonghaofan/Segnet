""" test.py
    apply to test no label
"""

import os
import sys
import time
import math
import logging
import importlib
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import make_dirs, save_img, vis_semseg
import cv2
flags = tf.app.flags

# Basic arguments
flags.DEFINE_string('arch', 'segnet', 'Network architecure')
flags.DEFINE_string('outdir', 'output/camvid', 'Output directory')
flags.DEFINE_string('resdir', 'test_results', 'Directory to visualize prediction')
flags.DEFINE_string('indir', 'test_data/test', 'Dataset directory')

# Dataset arguments
flags.DEFINE_string('dataset', 'camvid', 'Dataset name')
flags.DEFINE_string('model', 'Model','Directory where to read model checkpoint.')

# Evaluation arguments
flags.DEFINE_integer('channel', 3, 'Channel of an input image')
flags.DEFINE_integer('num_class', 5, 'Number of class to classify')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('height', 512, 'Input height')
flags.DEFINE_integer('width', 512, 'Input width')
flags.DEFINE_integer('num_sample', 755, 'Number of test samples')
FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def build_model(images, phase_train):
    model = importlib.import_module(FLAGS.arch)
    logits = model.inference(images, phase_train)
    prob, pred = model.predict(logits)
    return prob,pred

def test(res_dir):
    graph = tf.Graph()
    with graph.as_default():
        dataset = importlib.import_module(FLAGS.dataset)
        label_info = dataset.label_info
        cmap = label_info['cmap']

        image_list = [os.path.join(FLAGS.indir, i) for i in os.listdir(FLAGS.indir)]

        raw_image = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.height, FLAGS.width, 3],
                                name='images')

        phase_train = tf.placeholder(tf.bool, name='phase_train')
        output,pred = build_model(raw_image, phase_train)
        print(output)
        print(pred)
        pred = tf.reshape(pred, [-1,])
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(FLAGS.model)
        saver.restore(sess, ckpt.model_checkpoint_path)
        # saver = tf.train.import_meta_graph('./Model/model0.ckpt.meta')
        # saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model))
        # X, mode = tf.get_collection('inputs')
        # print('X, mode=',X, mode)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i, image_path in enumerate(image_list):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (FLAGS.width, FLAGS.height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            _pred= sess.run(pred,feed_dict={raw_image: img,
                           phase_train: False,
                           })
            _pred=_pred.reshape(FLAGS.height,FLAGS.width,1)
            img_name = image_path.split('/')[-1]
            # save_img(img, os.path.join(res_dir, 'img'), img_name)
            vis_semseg(_pred, cmap, os.path.join(res_dir, 'pred'), img_name)
            message = '{}img,test:{}'.format(i,res_dir + '/' + img_name)
            print(message)
        coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    if not os.path.exists(FLAGS.resdir):
        os.mkdir(FLAGS.resdir)
    test(FLAGS.resdir)
