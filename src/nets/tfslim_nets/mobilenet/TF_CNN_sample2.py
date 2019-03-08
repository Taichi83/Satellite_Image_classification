import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from subprocess import check_output
from datetime import timedelta
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.datasets import load_EUROsatdata, onehot_encd, image_normalize_std
from utils.visualize import barplot_classes, sample_images
from utils.eval import eval_labels

import tensorflow as tf
# from nets.tfslim_nets.mobilenet import mobilenet_v2

# Parameters
SPLIT_RATE_TEST = 0.05
SPLIT_RATE_VALID = 0.01
RANDOM_STATE = 1
IMAGE_NORMALIZATION = True
EPOCHS = 10

class CNN(object):
    def __init__(self, x_train_tr, x_train_val, y_train_tr, y_train_val, EPOCHS=20, 
                  early_stopping=None, BATCH_SIZE_TR=256, BATCH_SIZE_VAL=None):
        """Prepare tensor and build graph including layers, loss, accuracy, optimaization
        Args:
             x_train_tr: Numpy array of images for training [-1, hight, width, dims]
             x_train_val: Numpy array of images for validation [-1, hight, width, dims]
             y_train_tr: Numpy array of training labels [-1, number of classes]
             y_train_val: Numpy array of training labels [-1, number of classes]
             EPOCHS: epochs [number]
             early_stopping: Apply early stopping before the end of epochs [True/Flase]
             BATCH_SIZE_TR: Batch size of training data [number]
             BATCH_SIZE_VAL: Batch size of validation data [number]
        Returns:
        """
        # input -> global varibales
        self.x_train_tr = x_train_tr
        self.x_train_val = x_train_val
        self.y_train_tr = y_train_tr
        self.y_train_val = y_train_val
        self.EPOCHS = EPOCHS
        self.early_stopping = early_stopping
        self.BATCH_SIZE_TR = BATCH_SIZE_TR
        self.BATCH_SIZE_VAL = BATCH_SIZE_VAL
        
        tf.reset_default_graph()
        
        # Batch
        if self.BATCH_SIZE_TR == None:
            self.BATCH_SIZE_TR = self.x_train_tr.shape[0]
        self.n_batches_tr = self.x_train_tr.shape[0] // self.BATCH_SIZE_TR

        if self.BATCH_SIZE_VAL == None:
            self.BATCH_SIZE_VAL = self.x_train_val.shape[0]
        self.n_batches_val = self.x_train_val.shape[0] // self.BATCH_SIZE_VAL
    
        with tf.name_scope("inputs") as scope:
            # Placeholders of input images:X and imput labels:y
            self.X, self.y = tf.placeholder(tf.float32, shape=[None, self.x_train_tr.shape[1], self.x_train_tr.shape[2], self.x_train_tr.shape[3]]), tf.placeholder(tf.float32, shape=[None, self.y_train_tr.shape[1]])
            # batch size 
            self.batch_size_tr = tf.placeholder(tf.int64)
            self.batch_size_val = tf.placeholder(tf.int64)
            
            # switch of training or validation
            self.is_training = tf.placeholder(tf.bool)

            # Use tf.data.Dataset
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y)).batch(self.batch_size_tr).repeat()
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y)).batch(self.batch_size_val).repeat()

            # Create one iterator and initialize it with different datasets
            self.iterator = tf.data.Iterator.from_structure(
                self.train_dataset.output_types, self.train_dataset.output_shapes)

            # Get next batch of feature and labels by this command
            self.features, self.labels = self.iterator.get_next()

            # Initializer for train_dataset
            self.train_init_op = self.iterator.make_initializer(self.train_dataset)
            # Initializer for validatoin_dataset
            self.val_init_op = self.iterator.make_initializer(self.val_dataset)
        
        with tf.name_scope("model") as scope:
            self.conv1 = tf.layers.conv2d(inputs=self.features, filters=32, kernel_size=[5, 5], 
                                     padding="same", activation=tf.nn.relu)
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2,2], strides=2)

            self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=64, kernel_size=[5, 5], 
                                     padding="same", activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2,2], strides=2)    
            self.pool2_flat = tf.reshape(self.pool2, [-1, self.pool2.shape[1]*self.pool2.shape[2]*self.pool2.shape[3]])

            self.dense1 = tf.layers.dense(inputs=self.pool2_flat, units=1024, activation=tf.nn.relu)
            self.dropout = tf.layers.dropout(inputs=self.dense1, rate=0.5, training=self.is_training)

            self.dense2 = tf.layers.dense(inputs=self.dropout, units=256, activation=tf.nn.relu)

            self.logits = tf.layers.dense(inputs=self.dense2, units=self.y_train_tr.shape[1])
        
        with tf.name_scope("loss_func") as scope:
            # Loss function
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.logits)            
            self.v_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.logits)
            
            # for tensorboard
            self.train_summary_loss = tf.summary.scalar("softmax_cross_entropy", self.loss)
            self.val_summary_loss = tf.summary.scalar("val_softmax_cross_entropy", self.v_loss)

        with tf.name_scope("prediction") as scope:
            # Prediction
            self.prediction = tf.argmax(input=self.logits, axis=1)
            self.correct = tf.equal(tf.cast(self.prediction, tf.int32), 
                                    tf.cast(tf.argmax(input=self.labels, axis=1), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
            
            self.v_prediction = tf.argmax(input=self.logits, axis=1)
            self.v_correct = tf.equal(tf.cast(self.v_prediction, tf.int32), 
                                      tf.cast(tf.argmax(input=self.labels, axis=1), tf.int32))
            self.v_accuracy = tf.reduce_mean(tf.cast(self.v_correct, tf.float32))
            
            # for tensorboard
            self.train_summary_acc = tf.summary.scalar("accuracy", self.accuracy)
            self.val_summary_acc = tf.summary.scalar("val_accuracy", self.v_accuracy)
    
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate=1e-04, global_step=self.global_step, 
                                               decay_steps=10000, decay_rate=0.96, 
                                               staircase=True, 
                                               name="exp_decay")

        with tf.name_scope("optimizer") as scope:
            # Optimization
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(loss=self.loss,
                                          global_step=tf.train.get_global_step())
        
    def train(self, SAVE_DIR = '../tmp/models/tf_cnn', TB_DIR = '../tensorboard/tf_cnn', 
              PLOT_DIR = '../tmp/figures/tf_cnn'):
        """Execute training and validation
        Args:
            SAVE_DIR : String path to a folder for saving models
            TB_DIR: String path to a folder for saving tensorboards
            PLOT_DIR: String path to a folder for saving plot images
        Returns:
            tot_loss_list: list of training loss in each epoch
            tot_loss_val_list: list of validation loss in each epoch
            tot_acc_list: list of training accuracy in each epoch
            tot_acc_val_list: list of validation accuracy in each epoch
        """
        # Make dir if not exit.
        ## for saver
        if not os.path.isdir(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        ##  for tensorboard
        if not os.path.isdir(TB_DIR):
            os.makedirs(TB_DIR)
        ##  for plot images
        if not os.path.isdir(PLOT_DIR):
            os.makedirs(PLOT_DIR)
        
        # Create a saver.
        saver = tf.train.Saver()
        
        # Launch the graph and train and validate, saving the model.
        self.sess = tf.Session()
        
        # Initialize variables.
        self.sess.run(tf.global_variables_initializer())

        print('Training.....')

        # Start-time used for printing time-usage below.
        start_time = time.time()
        best_val_loss = float("inf")

        # Write summary in tensorboard format
        writer = tf.summary.FileWriter(TB_DIR, self.sess.graph)
        
        tot_loss_list = []
        tot_acc_list = []
        tot_loss_val_list = []
        tot_acc_val_list = []
        #  Exectue training and validation in each epoch
        for i in range(self.EPOCHS):
            tot_loss = 0.0
            tot_acc = 0.0
            tot_loss_val = 0.0
            tot_acc_val = 0.0

            # Execute training.
            # Set training data in Numpy array and batch size to Iterator.
            self.sess.run(self.train_init_op, feed_dict={self.X: self.x_train_tr, self.y: self.y_train_tr, 
                                               self.batch_size_tr: self.BATCH_SIZE_TR}) 
            # Calculation in every batch.
            for j in range(self.n_batches_tr):
                _, loss_value, acc, tr_smr_loss, tr_smr_acc = self.sess.run([self.train_op, self.loss, 
                                                    self.accuracy, self.train_summary_loss, self.train_summary_acc],
                                                             feed_dict={self.is_training: True})
                # Write summay of training loss and accracy to tensorboard.
                writer.add_summary(tr_smr_loss, j+i*self.n_batches_tr)
                writer.add_summary(tr_smr_acc, j+i*self.n_batches_tr)
                
                tot_loss += loss_value
                tot_acc += acc

            # Execute validation.
            # Set validation data in Numpy array and batch size to Iterator.
            self.sess.run(self.val_init_op, feed_dict={self.X: self.x_train_val, self.y: self.y_train_val, 
                                             self.batch_size_val: self.BATCH_SIZE_VAL}) 

            for j in range(self.n_batches_val):
                loss_value, acc, val_smr_loss, val_smr_acc = self.sess.run([self.v_loss, 
                                                    self.v_accuracy, self.val_summary_loss, self.val_summary_acc],
                                                             feed_dict={self.is_training: False})
                # Write summay of validation loss and accracy to tensorboard.
                writer.add_summary(val_smr_loss, j+i*self.n_batches_tr)
                writer.add_summary(val_smr_acc, j+i*self.n_batches_tr)
    
                tot_loss_val += loss_value
                tot_acc_val += acc

            print("Iter: {}, Loss: {:.4f}, Acc: {:4f}, Val_Loss: {:.4f}, Val_Acc: {:4f}".format(i, tot_loss/self.n_batches_tr, tot_acc/self.n_batches_tr, tot_loss_val/self.n_batches_val, tot_acc_val/self.n_batches_val))
            # Save models in each epoch
            saver.save(self.sess, os.path.join(SAVE_DIR, "model.ckpt"), i)
            
            tot_loss_list.append(tot_loss/self.n_batches_tr)
            tot_acc_list.append(tot_acc/self.n_batches_tr)
            tot_loss_val_list.append(tot_loss_val/self.n_batches_val)
            tot_acc_val_list.append(tot_acc_val/self.n_batches_val)

            if self.early_stopping:    
                    if tot_loss_val < best_val_loss:
                        best_val_loss = tot_loss_val
                        patience = 0
                    else:
                        patience += 1

                    if patience == self.early_stopping:
                        break
        
        # Plot training and validation loss
        fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12,4))

        axL.plot(range(len(tot_loss_list)), tot_loss_list, linestyle='dashed', linewidth = 1.0, marker='o', label='Training loss')
        axL.plot(range(len(tot_loss_val_list)), tot_loss_val_list, color='orange',  linestyle='solid', linewidth = 1.0, marker='o', label='Validation loss')
        axL.set_title('Training and validation loss')
        axL.legend()
        # plt.savefig(save_path_p1)
        axR.plot(range(len(tot_acc_list)), tot_acc_list, linestyle='dashed', linewidth = 1.0, marker='o', label='Training accuracy')
        axR.plot(range(len(tot_acc_val_list)), tot_acc_val_list, color='orange',  linestyle='solid', linewidth = 1.0, marker='o', label='Validation accuracy')
        axR.set_title('Training and validation accuracy')
        axR.legend()
        fig.savefig(os.path.join(PLOT_DIR, 'plot_loss_acc.png'), dpi=300)
        
        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))
        writer.close()

        return self.sess, tot_loss_list, tot_loss_val_list, tot_acc_list, tot_acc_val_list
    
    def test(self, x_test, y_test, BATCH_SIZE_TS=None):
        """Prepare tensor and build graph including layers, loss, accuracy, optimaization
        Args:
             x_test: Numpy array of images for training [-1, hight, width, dims]
             y_test: Numpy array of training labels [-1, number of classes]
             BATCH_SIZE_TS: Batch size of test data [number]
        Returns:
        """
        
        if BATCH_SIZE_TS == None:
            BATCH_SIZE_TS = x_test.shape[0]
        # n_batches_ts = x_test.shape[0] // BATCH_SIZE_TS
        
        batch_size_ts = tf.placeholder(tf.int64)
        with tf.name_scope("inputs") as scope:
            ts_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y)).batch(batch_size_ts).repeat()
            ts_init_op = self.iterator.make_initializer(ts_dataset)
        self.sess.run(ts_init_op, feed_dict={self.X: x_test, self.y: y_test, batch_size_ts: BATCH_SIZE_TS}) 
        pred = self.sess.run([self.v_prediction], feed_dict={self.is_training: True})
        return pred[0]
    
# read data
data_path="../datasets/EuroSAT_RGB/"
images, labels = load_EUROsatdata(data_path)
print(images.shape, labels.shape)

# one-hot encoding
labels_oh, cl_nms = onehot_encd(labels)
print('labels:', cl_nms)
print('Dimensionality of one-hot encoded label file:', labels_oh.shape)

# save bar-plot
barplot_classes(labels_oh, cl_nms, save_path="../tmp/figures/tf_cnn", filename='class_dist.png')

# visualize sample images
sample_images(images, labels_oh, cl_nms, save_path="../tmp/figures/tf_cnn", filename='sample_images.png')

# Split training and test data
x_train, x_test, y_train, y_test = train_test_split(
        images, labels_oh, test_size=SPLIT_RATE_TEST, stratify=labels_oh, random_state=RANDOM_STATE)

# Normalization
if IMAGE_NORMALIZATION:
    x_train, x_test = image_normalize_std(x_train, x_test)

# Split training and validation data
x_train_tr, x_train_val, y_train_tr, y_train_val = train_test_split(
        x_train, y_train, test_size=SPLIT_RATE_VALID, stratify=y_train, random_state=RANDOM_STATE)

print("Size of:")
print("- Training-set:\t\t{}".format(x_train_tr.shape[0]))
print("- Validation-set:\t\t{}".format(x_train_val.shape[0]))
print("- Test-set:\t\t{}".format(x_test.shape[0]))
print("Dim of:")
print("- Training-set:\t\t{}".format(x_train_tr.shape[1:4]))
print("- Validation-set:\t\t{}".format(x_train_val.shape[1:]))
print("- Test-set:\t\t{}".format(x_test.shape[1:]))

# Execute training and validation
model = CNN(x_train_tr, x_train_val, y_train_tr, y_train_val, EPOCHS=EPOCHS)
sess, tot_loss_list, tot_loss_val_list, tot_acc_list, tot_acc_val_list = model.train()

# Predict labels of test sets with trained model
pred_test = model.test(x_test, y_test)
y_test_le = np.argmax(y_test, axis=1)
eval_labels(y_test_le, pred_test, save_path="../tmp/calc_log/tf_cnn", filename='confusion_matrix.csv')