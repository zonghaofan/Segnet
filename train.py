#coding:utf-8
import os
import sys
import time
import logging
import importlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from datetime import datetime

from utils import make_dirs
from inputs import read_and_decode
from data_prepare import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# Basic arguments
flags.DEFINE_string('arch', 'segnet', 'Network architecure')
# Dataset arguments
flags.DEFINE_string('dataset', 'camvid', 'Dataset name')
flags.DEFINE_string('model', 'Model','Directory where to read model checkpoint.')
# Model arguments
flags.DEFINE_integer('channel', 3, 'Channel of an input image')
flags.DEFINE_integer('num_class', 5, 'Number of class to classify')
flags.DEFINE_integer('height', 256, 'Input height')
flags.DEFINE_integer('width', 256, 'Input width')

# Training arguments
flags.DEFINE_integer('epoch', 500, 'Epoch')
flags.DEFINE_integer('batch_size',8, 'Batch size')
# flags.DEFINE_integer('iteration', 50000, 'Number of training iterations')
flags.DEFINE_integer('num_threads', 8, 'Number of threads to read batches')
flags.DEFINE_integer('min_after_dequeue', 10, 'min_after_dequeue')
flags.DEFINE_integer('seed', 1234, 'Random seed')
flags.DEFINE_integer('snapshot', 2000, 'Snapshot')
flags.DEFINE_integer('print_step', 1, 'Number of step to print training log')
flags.DEFINE_string('optimizer', 'adam', 'optimizer')
# flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_boolean('cb', False, 'Class Balancing')
flags.DEFINE_string('data_dir','./data','dataset dir')
flags.DEFINE_string('log_dir','./logs','dataset dir')
np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

def set_config():
    #控制使用率
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
    gpu_options.allow_growth = True
    config = tf.ConfigProto(gpu_options=gpu_options)
    return config
def visualize_labels(labels):
    labels=tf.squeeze(labels)
    # labels = tf.cast(labels[..., 0], tf.int32)
    print(labels)
    table = tf.constant([[0, 0, 0],
                         [128, 0, 0],
                         [128, 128, 0],
                         [0, 128, 0],
                         [0, 0, 128]], tf.int32)
    out = tf.nn.embedding_lookup(table, labels)
    print(out)
    out = tf.cast(out, tf.float32)
    return out
def train():
    tf_config=set_config()
    graph = tf.Graph()
    with graph.as_default():
        dataset = importlib.import_module(FLAGS.dataset)

        train_imgs_dir, train_labels_dir = get_file_names(FLAGS.data_dir, 'train')
        num_train=len(train_imgs_dir)

        val_imgs_dir, val_labels_dir = get_file_names(FLAGS.data_dir, 'val')
        num_val = len(val_imgs_dir)
        train_imgs, train_labels = get_data_label_batch(train_imgs_dir, train_labels_dir, augmen=True)
        print(train_imgs)
        print(train_labels)
        val_imgs, val_labels = get_data_label_batch(val_imgs_dir, val_labels_dir, augmen=False)
    #
        images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, FLAGS.height, FLAGS.width, 3],
                                name='images')
        labels = tf.placeholder(dtype=tf.int32, shape=[FLAGS.batch_size, FLAGS.height, FLAGS.width, 1],
                                name='labels')
        phase_train = tf.placeholder(tf.bool, name='phase_train')
        print('images=', images)
        print('labels=', labels)

        tf.add_to_collection('inputs', images)
        tf.add_to_collection('inputs', phase_train)

        labels_vis=visualize_labels(labels)
        tf.summary.image('images', images, max_outputs=FLAGS.batch_size)
        tf.summary.image('labels', tf.cast(labels_vis, tf.float32), max_outputs=FLAGS.batch_size)

        global_steps = tf.train.get_or_create_global_step()

        cycle_rate=8*(num_train//FLAGS.batch_size)
        cycle = tf.cast(tf.floor(1. + tf.cast(global_steps, dtype=tf.float32) / (2 * cycle_rate)), dtype=tf.float32)

        x = tf.cast(tf.abs(tf.cast(global_steps, dtype=tf.float32) / cycle_rate - 2. * cycle + 1.), dtype=tf.float32)
        # learning_rate = 1e-6 + (1e-2 - 1e-6) * tf.maximum(0., (1 - x)) / tf.cast(2 ** (cycle - 1), dtype=tf.float32)
        learning_rate = 1e-3 + (1e-2 - 1e-3) * tf.maximum(0., (1 - x))
        model = importlib.import_module(FLAGS.arch)
        logits = model.inference(images, phase_train)
        prob, pred,red_ratio,yellow_ratio,green_ratio,blue_ratio = model.predict(logits,labels)
        print('prob.shape',prob)
        print('pred=', pred)
        print('logits=', logits)
        f1_score = model.f1_scores(prob, labels)
        pred_vis=visualize_labels(pred)
        tf.summary.image('pred', pred_vis, max_outputs=FLAGS.batch_size)

        if FLAGS.cb:
            iou_loss,loss_total = model.loss(logits, labels, cb=dataset.label_info['cb'])
        else:
            iou_loss,loss_total= model.loss(logits, labels)

        summary = model.setup_summary(iou_loss,loss_total,red_ratio,yellow_ratio,green_ratio,blue_ratio,learning_rate,f1_score)

        train_op = model.train_op(loss_total, FLAGS.optimizer, global_steps,
                                  lr=learning_rate, momentum=FLAGS.momentum)

    with tf.Session(graph=graph,config=tf_config) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train', sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.model) and tf.train.checkpoint_exists(FLAGS.model):
            latest_check_point = tf.train.latest_checkpoint(FLAGS.model)
            saver.restore(sess, latest_check_point)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch in range(FLAGS.epoch):
                for step in range(0, num_train, FLAGS.batch_size):
                    train_imgs_batch, train_labels_batch = sess.run([train_imgs, train_labels])
                    print('train_imgs_batch=',train_imgs_batch.shape)
                    print('train_labels_batch=',train_labels_batch.shape)
                    feed_dict = {
                                 images: train_imgs_batch,
                                 labels: train_labels_batch,
                                 phase_train: True}
                    _, loss_value,loss_iou ,f1_score_value, summary_str, global_step_value = sess.run(
                        [train_op, loss_total,iou_loss,f1_score, summary, global_steps], feed_dict=feed_dict)

                    print('train_loss={},train_f1_score={},train_step={}'.format(loss_value,f1_score_value,global_step_value))

                    train_writer.add_summary(summary_str, global_step_value)
        #
                for step in range(0, num_val, FLAGS.batch_size):
                    val_imgs_batch, val_labels_batch = sess.run([val_imgs, val_labels])
                    print('val_imgs_batch=',val_imgs_batch.shape)
                    print('val_labels_batch=', val_labels_batch.shape)
                    feed_dict = {images: val_imgs_batch,
                                 labels: val_labels_batch,
                                 phase_train: False}
                    _, loss_value,loss_iou, f1_score_value, summary_str=sess.run(
                        [train_op, loss_total,iou_loss,f1_score, summary], feed_dict=feed_dict)

                    val_step = epoch * (
                            num_train // FLAGS.batch_size) + step // FLAGS.batch_size * num_train // num_val

                    print('val_loss={},val_f1_score={},val_step={}'.format(loss_value, f1_score_value, val_step))

                    val_writer.add_summary(summary_str, val_step)
                if epoch % 1 == 0:
                    saver.save(sess, "Model/model{}.ckpt".format(epoch))
                    print('save model{}'.format(epoch))
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "./Model/model.ckpt")
if __name__ == '__main__':
    train()
