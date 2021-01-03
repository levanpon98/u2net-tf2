import os
import tqdm

from absl import app, flags
from absl.flags import FLAGS
import tensorflow as tf

from models.u2net import u2net
from models.losses import bce_loss
from dataloader import Dataset

flags.DEFINE_string('data_path', default='/home/ponlv/work/data/CelebAMask-HQ', help='Path of dataset')
flags.DEFINE_string('model_path', default='./saved_model', help='Path of model')
flags.DEFINE_integer('epochs', default=10, help='Epoch')
flags.DEFINE_integer('batch_size', default=8, help='Batch size')
flags.DEFINE_integer('image_size', default=288, help='Image size')
flags.DEFINE_float('learning_rate', default=0.001, help='Init learning rate')
flags.DEFINE_bool('use_gpu', default=False, help='Use GPU for training if `use_gpu` is True')


# flags.DEFINE_integer('num_gpu', default=1, help='Number of GPU are using for training with GPU')


def main(_):
    devices = '/cpu:0' if FLAGS.use_gpu is False else '/gpu:0'

    datasets = Dataset(data_path=FLAGS.data_path, batch_size=FLAGS.batch_size)
    num_step = len(datasets) // FLAGS.batch_size

    # Define graph
    with tf.device(devices):
        graph = tf.Graph()
        with graph.as_default():
            X = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 3])
            Y = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, 1])

            logis_train = u2net(X, 1)
            loss_op = bce_loss(Y, logis_train)
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            grads = optimizer.compute_gradients(loss_op)
            train_op = optimizer.apply_gradients(grads)
            summary_loss = tf.summary.scalar("loss", tf.reduce_mean(loss_op))

            initialize = tf.global_variables_initializer()

            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:

            sess.run(initialize)
            summary_writer = tf.summary.FileWriter('./logs')
            for epoch in range(FLAGS.epochs):
                for step in tqdm.tqdm(range(0, num_step)):
                    batch_x, batch_y = datasets[step]

                    _, loss, summary = sess.run([train_op, loss_op, summary_loss], feed_dict={X: batch_x,
                                                                                              Y: batch_y})
                    summary_writer.add_summary(summary, epoch * num_step + step)
            # Save model weights to disk
            save_path = saver.save(sess, FLAGS.model_path)
            print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    app.run(main)
