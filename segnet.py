""" segnet.py
    Implementation of SegNet for Semantic Segmentation.
"""


import os
import sys
import time
import numpy as np
import tensorflow as tf

from ops import *

flags = tf.app.flags
FLAGS = flags.FLAGS

mean_RGB=[123.68,116.779,103.939]

def inference(inputs, phase_train):
    inputs=inputs-mean_RGB
    R=inputs[...,-3]/58.393
    G = inputs[..., -2] / 57.12
    B = inputs[..., -1] / 57.375
    inputs=tf.concat((tf.expand_dims(R,axis=-1),tf.expand_dims(G,axis=-1),tf.expand_dims(B,axis=-1)),axis=-1)
    with tf.variable_scope(FLAGS.arch):
        h, mask = encoder(inputs, phase_train, name='encoder')
        print('h, mask',h, mask)
        logits = decoder(h, mask, phase_train, name='decoder')
    return logits

def area_loss(label,logit):
    area_loss=tf.cond(tf.reduce_mean(label)< 1e-6,lambda :0.,lambda:(1. - tf.reduce_mean(tf.square(logit)) / tf.reduce_mean(tf.square(label))))
    # if tf.reduce_mean(label) < 1e-6:
    #     area_loss = tf.constant(0.)
    # else:
    #     area_loss = 1. - tf.reduce_mean(logit) / tf.reduce_mean(label)
    area_loss=tf.cond(area_loss<-1e-6,lambda:1.,lambda: area_loss)
    return area_loss

def loss(logits, labels, cb=None, name='loss'):
    with tf.name_scope(name):
        num_class = logits.get_shape().as_list()[-1]
        epsilon = tf.constant(value=1e-10)

        labels_=tf.one_hot(tf.squeeze(labels),depth=num_class)
        inter = tf.reduce_sum(tf.multiply(labels_, logits))
        union = tf.add(tf.reduce_sum(tf.square(labels_)), tf.reduce_sum(tf.square(logits)))
        m_IOU_loss=1 - 2 * (inter + 1) / (union + 1)

        # red_area_loss=area_loss(labels_[...,-4],logits[...,-4])
        # yellow_area_loss =area_loss(labels_[...,-3],logits[...,-3])
        # green_area_loss =area_loss(labels_[...,-2],logits[...,-2])
        # blue_area_loss =area_loss(labels_[...,-1],logits[...,-1])

        # logits = tf.reshape(logits, (-1, num_class))
        # labels = tf.reshape(labels, (-1, 1))
        # not_ign_mask = tf.where(tf.not_equal(tf.squeeze(labels), ignore_label))
        #
        # logits = tf.reshape(tf.gather(logits, not_ign_mask), (-1, num_class))
        # labels = tf.reshape(tf.gather(labels, not_ign_mask), (-1, 1))
        #
        # one_hot = tf.reshape(
        #     tf.one_hot(labels, depth=num_class), (-1, num_class))
        #
        # prob = tf.nn.softmax(logits)

        if cb is not None:
            xe = -tf.reduce_sum(
                tf.multiply(labels_ * tf.log(logits + epsilon), cb),
                reduction_indices=[1])
        else:
            xe = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels_, logits=logits)

        loss_total = tf.reduce_mean(xe)+m_IOU_loss#+\
                     #tf.reduce_mean(red_area_loss+yellow_area_loss+green_area_loss+blue_area_loss)
    return m_IOU_loss,loss_total#,red_area_loss,yellow_area_loss,green_area_loss,blue_area_loss


def f1_scores(prob, labels,name='f1_scores'):
    with tf.name_scope(name):
        num_class = prob.get_shape().as_list()[-1]
        labels=tf.one_hot(labels,depth=num_class)
        mean_f1_score_list=[]
        for i in range(num_class):
            pred=prob[...,i]
            label=labels[...,i]
            pred=tf.reshape(pred,[-1])
            label=tf.reshape(label,[-1])
            TP=pred*label
            FP=(1.-label)*pred
            FN=(1.-pred)*label
            precision=tf.reduce_sum(TP) / (tf.reduce_sum(TP)+tf.reduce_sum(FP) + 1e-7)
            recall = tf.reduce_sum(TP) / (tf.reduce_sum(TP) + tf.reduce_sum(FN) + 1e-7)
            f1_score = 2 / (1 / precision + 1 / recall + 1e-7)
            mean_f1_score_list.append(f1_score)
        mean_f1_score=tf.reduce_mean(mean_f1_score_list)
        # correct_pred=tf.equal(pred,labels)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return mean_f1_score

        # logits = tf.reshape(logits, (-1, FLAGS.num_class))
        # labels = tf.reshape(labels, [-1])
        # print('labels=',labels )
        # not_ign_mask = tf.where(tf.not_equal(tf.squeeze(labels), ignore_label))
        # print('not_ign_mask=',not_ign_mask)
        # print('logits=',logits)
        # print('tf.gather(logits, not_ign_mask)=',tf.gather(logits, not_ign_mask))
        # logits = tf.reshape(tf.gather(logits, not_ign_mask), (-1, FLAGS.num_class))
        #
        # labels = tf.reshape(tf.gather(labels, not_ign_mask), [-1])
        #
        # epsilon = tf.constant(value=1e-10, name="epsilon")
        # logits = tf.add(logits, epsilon)
        #
        # prob = tf.nn.softmax(logits)
        # pred = tf.cast(tf.argmax(prob, axis=1), tf.int32)
        #
        # correct_pred = tf.equal(pred, labels)
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # # pred=tf.reshape(pred,(-1,224,224,1))
        # return accuracy
def area_ratio(pred,label,index):
    ratio=tf.cond(tf.reduce_sum(tf.cast(tf.equal(label, index), tf.float32))< 1e-6,lambda:1.,\
          lambda:tf.reduce_sum(tf.cast(tf.equal(pred, index), tf.float32))/(tf.reduce_sum(tf.cast(tf.equal(label, index), tf.float32)) + 1e-6))
    return ratio


def predict(logits, labels,name='predict'):
    with tf.name_scope(name):
        prob = tf.squeeze(tf.nn.softmax(logits)tf.nn.softmax(logits))
        pred = tf.squeeze(tf.cast(tf.argmax(prob, axis=-1), tf.int32))
        label=tf.squeeze(labels)
        red_ratio=area_ratio(pred,label,index=1)
        yellow_ratio = area_ratio(pred, label,index=2)
        green_ratio = area_ratio(pred, label,index=3)
        blue_ratio = area_ratio(pred, label,index=4)
        # red_ratio=tf.reduce_sum(tf.cast(tf.equal(pred,1),tf.float32))/(tf.reduce_sum(tf.cast(tf.equal(labels,1),tf.float32))+1e-6)
        # yellow_ratio = tf.reduce_sum(tf.cast(tf.equal(pred, 2),tf.float32)) / tf.reduce_sum(tf.cast(tf.equal(labels, 2),tf.float32)+1e-6)
        # green_ratio = tf.reduce_sum(tf.cast(tf.equal(pred, 3),tf.float32)) / tf.reduce_sum(tf.cast(tf.equal(labels, 3),tf.float32)+1e-6)
        # blue_ratio = tf.reduce_sum(tf.cast(tf.equal(pred,4),tf.float32)) / tf.reduce_sum(tf.cast(tf.equal(labels, 4),tf.float32)+1e-6)
    return prob, pred,red_ratio,yellow_ratio,green_ratio,blue_ratio


def train_op(loss, opt_name,steps, **kwargs):
    optimizer = _get_optimizer(opt_name, kwargs)
    return optimizer.minimize(loss,global_step=steps)


def setup_summary(iou_loss,total_loss,red_area,yellow_area,green_area,blue_area,lr ,f1_score):
    tf.summary.scalar('iou_loss', iou_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('red_area_ratio', red_area)
    tf.summary.scalar('yellow_area_ratio', yellow_area)
    tf.summary.scalar('green_area_ratio', green_area)
    tf.summary.scalar('blue_area_ratio', blue_area)
    tf.summary.scalar('learning rate',lr)
    tf.summary.scalar('f1_score', f1_score)
    return tf.summary.merge_all()#tf.summary.merge([summary_loss, summary_acc])


def _get_optimizer(opt_name, params):
    if opt_name == 'adam':
        return tf.train.AdamOptimizer(params['lr'])
    elif opt_name == 'adadelta':
        return tf.train.AdadeltaOptimizer(params['lr'])
    elif opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(params['lr'])
    elif opt_name == 'momentum':
        return tf.train.MomentumOptimizer(params['lr'], params['momentum'])
    elif opt_name == 'rms':
        return tf.train.RMSPropOptimizer(params['lr'])
    elif opt_name == 'adagrad':
        return tf.train.AdagradOptimizer(params['lr'])
    else:
        print('error')


def n_enc_block(inputs, phase_train, n, k, name):
    h = inputs
    with tf.variable_scope(name):
        for i in range(n):
            h = conv2d(h, k, 3, stride=1, name='conv_{}'.format(i + 1))
            h = batch_norm(h, phase_train, name='bn_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
        h, mask = maxpool2d_with_argmax(h, name='maxpool_{}'.format(i + 1))
    return h, mask


def encoder(inputs, phase_train, name='encoder'):
    with tf.variable_scope(name):
        print('inputs=',inputs)
        h, mask_1 = n_enc_block(inputs, phase_train, n=2, k=64, name='block_1')
        print('h, mask_1',h, mask_1)
        h, mask_2 = n_enc_block(h, phase_train, n=2, k=128, name='block_2')
        print('h, mask_2', h, mask_2)
        h, mask_3 = n_enc_block(h, phase_train, n=3, k=256, name='block_3')
        print('h, mask_3', h, mask_3)
        h, mask_4 = n_enc_block(h, phase_train, n=3, k=512, name='block_4')
        print('h, mask_4', h, mask_4)
        h, mask_5 = n_enc_block(h, phase_train, n=3, k=512, name='block_5')
        print('h, mask_5', h, mask_5)
    return h, [mask_5, mask_4, mask_3, mask_2, mask_1]


def n_dec_block(inputs, mask, adj_k, phase_train, n, k, name):
    # in_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name):
        h = maxunpool2d(inputs, mask, name='unpool')
        for i in range(n):
            if i == (n - 1) and adj_k:
                h = conv2d(h, k / 2, 3, stride=1, name='conv_{}'.format(i + 1))
            else:
                h = conv2d(h, k, 3, stride=1, name='conv_{}'.format(i + 1))
            h = batch_norm(h, phase_train, name='bn_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
    return h


def dec_last_conv(inputs, phase_train, k, name):
    with tf.variable_scope(name):
        h = conv2d(inputs, k, 1, name='conv')
    return h


def decoder(inputs, mask, phase_train, name='decoder'):
    with tf.variable_scope(name):
        h = n_dec_block(inputs, mask[0], False, phase_train, n=3, k=512, name='block_5')
        print('h=',h)
        print('mask[0]=',mask[0])
        h = n_dec_block(h, mask[1], True, phase_train, n=3, k=512, name='block_4')
        print('h=', h)
        print('mask[1]=', mask[1])
        h = n_dec_block(h, mask[2], True, phase_train, n=3, k=256, name='block_3')
        print('h=', h)
        print('mask[2]=', mask[2])
        h = n_dec_block(h, mask[3], True, phase_train, n=2, k=128, name='block_2')
        print('h=', h)
        print('mask[3]=', mask[3])
        h = n_dec_block(h, mask[4], True, phase_train, n=2, k=64, name='block_1')
        print('h=', h)
        print('mask[4]=', mask[4])
        h = dec_last_conv(h, phase_train, k=FLAGS.num_class, name='last_conv')
        print('h=', h)
    logits = h
    return logits
