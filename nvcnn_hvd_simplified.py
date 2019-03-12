#!/usr/bin/env python
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import print_function
from builtins import range

import numpy as np
import tensorflow as tf

import sys
import os
import time
import math
from collections import defaultdict

try:
    import horovod.tensorflow as hvd
except:
    print("Failed to import horovod module. "
          "%s is intended for use with Uber's Horovod distributed training "
          "framework. To create a Docker image with Horovod support see "
          "docker-examples/Dockerfile.horovod." % __file__)
    raise

#    
from ImagePreprocessor import ImagePreprocessor
# Common routines used for generation of convolutional neural netowrks    
from GPUNetworkBuilder import GPUNetworkBuilder
    
from FeedForwardTrainer import FeedForwardTrainer

import settings
    

# ResNet implementation
from ResNet import inference_resnet_v1
# AlexNet implementation
from AlexNet import inference_alexnet_owt


hvd.init()

def print_r0(*args, **kwargs):
    '''This function will ensure that we print output only to the stdout of the main process.
    '''
    if hvd.rank() == 0:
        print(*args, **kwargs)



        
def main():
    tf.set_random_seed(1234)
    np.random.seed(4321)

    '''Since we will be executing our code using the command line we need to process the user input.
       This helper method hides the complexity of this process to simplify the code using in this class.
    '''
    FLAGS = settings.parseCmdLine()
    
    # ImageNet used contains 1000 classes.
    nclass = 1000
    # The batch size we will use during training.
    batch_size = FLAGS.batch_size
    # Subset of data to use (either training or validation).
    subset = 'train'

    print_r0("Cmd line args:")
    print_r0('\n'.join(['  '+arg for arg in sys.argv[1:]]))


    def get_num_records(tf_record_pattern):
        def count_records(tf_record_filename):
            count = 0
            for _ in tf.python_io.tf_record_iterator(tf_record_filename):
                count += 1
            return count
        filenames = sorted(tf.gfile.Glob(tf_record_pattern))
        nfile = len(filenames)
        return (count_records(filenames[0])*(nfile-1) +
                count_records(filenames[-1]))
    
    if FLAGS.data_dir is not None and FLAGS.data_dir != '':
        nrecord = get_num_records(os.path.join(FLAGS.data_dir, '%s-*' % subset))
    else:
        nrecord = FLAGS.num_batches * batch_size * hvd.size()

    
    FLAGS.fp16 = False
    
    # Training hyperparameters
    FLAGS.learning_rate         = 0.001 # Model-specific values are set below
    FLAGS.momentum              = 0.9
    FLAGS.lr_decay_policy       = 'step'
    FLAGS.lr_decay_epochs       = 30
    FLAGS.lr_decay_rate         = 0.1
    FLAGS.lr_poly_power         = 2.
    FLAGS.weight_decay          = 1e-4
    FLAGS.input_buffer_size     = min(10000, nrecord)
    FLAGS.distort_color         = False
    FLAGS.nstep_burnin          = 20
    FLAGS.summary_interval_secs = 600
    FLAGS.save_interval_secs    = 600

    model_dtype = tf.float16 if FLAGS.fp16 else tf.float32

    print_r0("Num ranks:  ", hvd.size())
    print_r0("Num images: ", nrecord if FLAGS.data_dir is not None else 'Synthetic')
    print_r0("Model:      ", FLAGS.model)
    print_r0("Batch size: ", batch_size, 'per device')
    print_r0("            ", batch_size * hvd.size(), 'total')
    print_r0("Data format:", 'NCHW')
    print_r0("Data type:  ", 'fp16' if model_dtype == tf.float16 else 'fp32')

    if FLAGS.num_epochs is not None:
        if FLAGS.data_dir is None:
            raise ValueError("num_epochs requires data_dir to be specified")
        nstep = nrecord * FLAGS.num_epochs // (batch_size * hvd.size())
    else:
        nstep = FLAGS.num_batches
        FLAGS.num_epochs = max(nstep * batch_size * hvd.size() // nrecord, 1)

    model_name = FLAGS.model

    

    if model_name == 'alexnet':
        height, width = 227, 227
        model_func = inference_alexnet_owt
        FLAGS.learning_rate = 0.03
    elif model_name.startswith('resnet'):
        height, width = 224, 224
        nlayer = int(model_name[len('resnet'):])
        model_func = lambda net, images: inference_resnet_v1(net, images, nlayer)
        FLAGS.learning_rate = 0.1 if nlayer > 18 else 0.5
    else:
        raise ValueError("Invalid model type: %s" % model_name)

        
    preprocessor = ImagePreprocessor(height, width, subset)

    
    def loss_func(images, labels, var_scope):
        # Build the forward model
        net = GPUNetworkBuilder(True, dtype=model_dtype)
        output = model_func(net, images)
        # Add final FC layer to produce nclass outputs
        logits = net.fully_connected(output, nclass, activation='LINEAR')
        if logits.dtype != tf.float32:
            logits = tf.cast(logits, tf.float32)
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)
        # Add weight decay
        if FLAGS.weight_decay is not None and FLAGS.weight_decay != 0.:
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=var_scope.name)
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in params])
            if l2_loss.dtype != tf.float32:
                l2_loss = tf.cast(l2_loss, tf.float32)
            loss += FLAGS.weight_decay * l2_loss
        return loss, logits

    nstep_per_epoch = nrecord // batch_size
    
    
    trainer = FeedForwardTrainer(preprocessor, loss_func, nstep_per_epoch)
    print_r0("Building training graph")
    total_loss, learning_rate, train_ops = trainer.training_step(
        batch_size)
    
    

    print_r0("Creating session")
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    sess = tf.Session(config=config)

    train_writer = None
    saver = None
    summary_ops = None
    
    

    if hvd.rank() == 0 and len(FLAGS.log_dir):
        log_dir = FLAGS.log_dir
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        summary_ops = tf.summary.merge_all()
        last_summary_time = time.time()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=3)
        last_save_time = time.time()


    print_r0("Initializing variables")
    trainer.init(sess)

    restored = False
    if hvd.rank() == 0 and saver is not None:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        checkpoint_file = os.path.join(log_dir, "checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            restored = True
            print("Restored session from checkpoint " + ckpt.model_checkpoint_path)
        else:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

    trainer.sync(sess)

    if hvd.rank() == 0 and not restored:
        if saver is not None:
            save_path = saver.save(sess, checkpoint_file, global_step=0)
            print("Checkpoint written to", save_path)

    

    print_r0("Pre-filling input pipeline")
    trainer.prefill_pipeline(sess)

    print_r0("Training")
    print_r0("  Step Epoch Img/sec   Loss   LR")
    batch_times = []
    oom = False
    step0 = int(sess.run(trainer.global_step))
    
    

    for step in range(step0, nstep):
        ops_to_run = [total_loss, learning_rate] + train_ops
        try:
            start_time = time.time()
            if (hvd.rank() == 0 and summary_ops is not None and
                (step == 0 or
                 time.time() - last_summary_time > FLAGS.summary_interval_secs)):
                if step != 0:
                    last_summary_time += FLAGS.summary_interval_secs
                print("Writing summaries to ", log_dir)
                summary, loss, lr = sess.run([summary_ops] + ops_to_run)[:3]
                train_writer.add_summary(summary, step)
            else:
                loss, lr = sess.run(ops_to_run)[:2]
            elapsed = time.time() - start_time
        except KeyboardInterrupt:
            print("Keyboard interrupt")
            break
        except tf.errors.ResourceExhaustedError:
            elapsed = -1.
            loss    = 0.
            lr      = -1
            oom = True

        if (hvd.rank() == 0 and saver is not None and
            time.time() - last_save_time > FLAGS.save_interval_secs):
            last_save_time += FLAGS.save_interval_secs
            save_path = saver.save(sess, checkpoint_file,
                                   global_step=trainer.global_step)
            print("Checkpoint written to", save_path)

        if step >= FLAGS.nstep_burnin:
            batch_times.append(elapsed)
        img_per_sec = batch_size / elapsed
        effective_accuracy = 100. / math.exp(min(loss,20.))
        if step == 0 or (step+1) % FLAGS.display_every == 0:
            epoch = step*batch_size*hvd.size() // nrecord
            print_r0("%6i %5i %7.1f %7.3f %7.5f" % (
                step+1, epoch+1, img_per_sec*hvd.size(), loss, lr))
        if oom:
            break
    

    nstep = len(batch_times)
    if nstep > 0:
        batch_times = np.array(batch_times)
        speeds = batch_size*hvd.size() / batch_times
        speed_mean = np.mean(speeds)
        if nstep > 2:
            speed_uncertainty = np.std(speeds, ddof=1) / np.sqrt(float(nstep))
        else:
            speed_uncertainty = float('nan')
        speed_madstd = 1.4826*np.median(np.abs(speeds - np.median(speeds)))
        speed_jitter = speed_madstd
        print_r0('-' * 64)
        print_r0('Images/sec: %.1f +/- %.1f (jitter = %.1f)' % (
            speed_mean, speed_uncertainty, speed_jitter))
        print_r0('-' * 64)
    else:
        print_r0("No results, did not get past burn-in phase (%i steps)" %
              FLAGS.nstep_burnin)

    if train_writer is not None:
        train_writer.close()

    if oom:
        print("Out of memory error detected, exiting")
        sys.exit(-2)

if __name__ == '__main__':
    main()
