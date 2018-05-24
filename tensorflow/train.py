import sys
sys.path.append("..")
import numpy as np
import os
from resnet import *
from cifar10 import *
from logger import *

gpu_number = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
    
class Trainer():
    def __init__(self, train_batch_size=256, test_batch_size=500, exp_name='exp'):
        self.model = ResNet([2, 2, 2, 2])
        self.dataset = Cifar10(train_batch_size, test_batch_size)
        self.image_shape = self.dataset.input_shape
        self.num_classes = self.dataset.num_classes
        self.build_graph()
        self.logger = Logger(exp_dir=exp_name)
        self.tensorboard_path = '/tmp/resnet/'+ exp_name

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # session config
            self.config = tf.ConfigProto(allow_soft_placement=True)
            self.config.gpu_options.allow_growth = True

            # input placeholders
            self.image_placeholder = tf.placeholder(
            	tf.float32,
            	shape = [None] + self.image_shape,
            	name = 'images'
            )
            self.label_placeholder = tf.placeholder(
                tf.int32,
                shape = [None, ],
                name = 'labels'
            )
            
            # network          
            self.logits = self.model.build_network(self.image_placeholder, self.num_classes)
               
            # loss
            cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                           labels = self.label_placeholder,
                                           logits = self.logits
                                      )) 
            #regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            #decay = tf.train.exponential_decay(0.0002, self.global_step, 480000, 0.2, staircase=True)
            #self.loss = cross_entropy_loss + decay * sum(regularization_loss)
            self.loss = cross_entropy_loss

            # probability
            self.probability = tf.nn.softmax(self.logits)

            # prediction
            self.prediction = tf.argmax(self.probability, axis=1)

            # accuracy
            correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.label_placeholder)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # self.accuracy = tf.reduce_mean(tf.to_float32(predictions == labels))
            # self.accuracy, _ = tf.metrics.accuracy(self.label_placeholder, self.prediction)

            # optimizer
            #lr_boundaries = [400, 32000, 48000, 64000]
            #lr_values = [0.01, 0.1, 0.01, 0.001, 0.0002]
            #learning_rate = tf.train.piecewise_constant(self.global_step, lr_boundaries, lr_values)
            #self.learning_rate = learning_rate
            self.learning_rate = tf.Variable(initial_value=0.1, trainable=False)

            train_vars = [x for x in tf.trainable_variables() if 'ResNet' in x.name]
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim = optimizer.minimize(self.loss, global_step=self.global_step, var_list=train_vars)

            # train summary
            # loss 
            self.train_loss = tf.placeholder(tf.float32, shape=())
            self.train_loss_summary = tf.summary.scalar('train_loss', self.train_loss)
            # acc 
            self.train_accuracy_value = tf.placeholder(tf.float32, shape=())
            self.train_accuracy_summary = tf.summary.scalar('train_accuracy', self.train_accuracy_value)
            
            # test summary
            # loss 
            self.test_loss = tf.placeholder(tf.float32, shape=())
            self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
            # acc 
            self.test_accuracy_value = tf.placeholder(tf.float32, shape=())
            self.test_accuracy_summary = tf.summary.scalar('test_accuracy', self.test_accuracy_value)

    def train(self, epochs=1):
        with self.graph.as_default():
            writer = tf.summary.FileWriter(self.tensorboard_path, graph=tf.get_default_graph())
            with tf.Session(config=self.config) as sess:
                saver = tf.train.Saver(max_to_keep=100)
                all_initializer_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(all_initializer_op)
                for i in range(epochs):
                    total_loss = 0.0
                    total_accuracy = 0.0
                    self.dataset.shuffle_dataset()
                    for j in range(self.dataset.train_batch_count):
                        batch_images, batch_labels = self.dataset.next_aug_train_batch(j)
                        loss, logits, accuracy,  _ = sess.run([self.loss, self.logits, self.accuracy, self.optim], 
                                                                 feed_dict = {self.image_placeholder : batch_images, 
                                                                              self.label_placeholder : batch_labels})
                        total_loss += loss
                        total_accuracy += accuracy
                    avg_loss = total_loss / self.dataset.train_batch_count 
                    avg_accuracy = total_accuracy / self.dataset.train_batch_count
                    # logging training results
                    summary = sess.run(self.train_loss_summary, feed_dict={self.train_loss: avg_loss})
                    writer.add_summary(summary, i)
                    summary = sess.run(self.train_accuracy_summary, feed_dict={self.train_accuracy_value: avg_accuracy})
                    writer.add_summary(summary, i) 
                    self.logger.log('Training epoch {0}, learning rate {1}'.format(i, sess.run(self.learning_rate)))
                    self.logger.log('    train loss {0}, train error {1}'.format(avg_loss, 1.0 - avg_accuracy))
                    total_loss = 0.0
                    total_accuracy = 0.0
                    # evaluate on test set
                    for j in range(self.dataset.test_batch_count)  :
                        batch_images, batch_labels = self.dataset.next_test_batch(j)
                        loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], 
                                                              feed_dict = {self.image_placeholder : batch_images, 
                                                                           self.label_placeholder : batch_labels})
                        total_loss += loss
                        total_accuracy += accuracy
                    avg_loss = total_loss / self.dataset.test_batch_count
                    avg_accuracy = total_accuracy / self.dataset.test_batch_count
                    # logging validation results
                    summary = sess.run(self.test_loss_summary, feed_dict={self.test_loss: avg_loss})
                    writer.add_summary(summary, i)
                    summary = sess.run(self.test_accuracy_summary, feed_dict={self.test_accuracy_value: avg_accuracy})
                    writer.add_summary(summary, i)
                    self.logger.log('    test loss {0}, test error {1}'.format(avg_loss, 1.0 - avg_accuracy))
                    # save model
                    save_model_file = os.path.join(self.logger.exp_dir, 'ResNet-model')
                    if i % 20 == 0:
                        saver.save(sess, save_model_file, global_step=self.global_step)

if __name__ == "__main__":
    trainer = Trainer(exp_name='exp6')
    trainer.train(epochs=400)
