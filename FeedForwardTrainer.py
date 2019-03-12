import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import settings

try:
    import horovod.tensorflow as hvd
except:
    print("Failed to import horovod module. "
          "%s is intended for use with Uber's Horovod distributed training "
          "framework. To create a Docker image with Horovod support see "
          "docker-examples/Dockerfile.horovod." % __file__)
    raise

class FeedForwardTrainer(object):
    def __init__(self, preprocessor, loss_func, nstep_per_epoch=None):
        self.image_preprocessor = preprocessor
        self.loss_func          = loss_func
        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0),
                dtype=tf.int64,
                trainable=False)
        if settings.FLAGS.lr_decay_policy == 'poly':
            self.learning_rate = tf.train.polynomial_decay(
                settings.FLAGS.learning_rate,
                self.global_step,
                decay_steps=settings.FLAGS.num_epochs*nstep_per_epoch,
                end_learning_rate=0.,
                power=settings.FLAGS.lr_poly_power,
                cycle=False)
        else:
            self.learning_rate = tf.train.exponential_decay(
                settings.FLAGS.learning_rate,
                self.global_step,
                decay_steps=settings.FLAGS.lr_decay_epochs*nstep_per_epoch,
                decay_rate=settings.FLAGS.lr_decay_rate,
                staircase=True)
    
    def __stage(self, tensors):
        """Stages the given tensors in a StagingArea for asynchronous put/get.
        """
        stage_area = data_flow_ops.StagingArea(
            dtypes=[tensor.dtype       for tensor in tensors],
            shapes=[tensor.get_shape() for tensor in tensors])
        put_op      = stage_area.put(tensors)
        get_tensors = stage_area.get()

        get_tensors = [tf.reshape(gt, t.get_shape())
                       for (gt,t) in zip(get_tensors, tensors)]
        return put_op, get_tensors

    def __float32_variable_storage_getter(self, getter, name, shape=None, dtype=None,
                                        initializer=None, regularizer=None,
                                        trainable=True,
                                        *args, **kwargs):
        storage_dtype = tf.float32 if trainable else dtype
        variable = getter(name, shape, dtype=storage_dtype,
                          initializer=initializer, regularizer=regularizer,
                          trainable=trainable,
                          *args, **kwargs)
        if trainable and dtype != tf.float32:
            variable = tf.cast(variable, dtype)
        return variable
            
    def training_step(self, batch_size):
        with tf.device('/cpu:0'):
            images, labels = self.image_preprocessor.minibatch(batch_size)
            # Stage images on the host
            preload_op, (images, labels) = self.__stage([images, labels])
        with tf.device('/gpu:0'):
            # Copy images from host to device
            gpucopy_op, (images, labels) = self.__stage([images, labels])

        with tf.device('/gpu:0'):
            # Evaluate the loss and compute the gradients
            with tf.variable_scope(
                    'GPU_0',
                    # Force all variables to be stored as float32
                    custom_getter=self.__float32_variable_storage_getter) as var_scope:
                loss, logits = self.loss_func(images, labels, var_scope)
                
                with tf.device('/cpu:0'): # No in_top_k implem on GPU
                    top1 = tf.reduce_mean(
                        tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))
                    top5 = tf.reduce_mean(
                        tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))

        with tf.device('/cpu:0'):
            averager = tf.train.ExponentialMovingAverage(0.90, name='loss_avg',
                                                         zero_debias=True)
            avg_op = averager.apply([loss])
            loss_avg = averager.average(loss)
            # Note: This must be done _after_ the averager.average() call
            #         because it changes total_loss into a new object.
            with tf.control_dependencies([avg_op]):
                total_loss     = tf.identity(loss)
                total_loss_avg = tf.identity(loss_avg)
            tf.summary.scalar('total loss raw', total_loss)
            tf.summary.scalar('total loss avg', total_loss_avg)
            tf.summary.scalar('train accuracy top-1 %', 100.*top1)
            tf.summary.scalar('train accuracy top-5 %', 100.*top5)
            tf.summary.scalar('learning rate', self.learning_rate)

        # Apply the gradients to optimize the loss function
        with tf.device('/gpu:0'):
            opt = tf.train.MomentumOptimizer(self.learning_rate, settings.FLAGS.momentum,
                                             use_nesterov=True)
            opt = hvd.DistributedOptimizer(opt)
            train_op = opt.minimize(loss,
                                    gate_gradients=tf.train.Optimizer.GATE_NONE)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []
        with tf.device('/cpu:0'):
            increment_global_step_op = tf.assign_add(self.global_step, 1)
        update_ops.append(increment_global_step_op)
        self.enqueue_ops = []
        self.enqueue_ops.append(preload_op)
        if gpucopy_op is not None:
            self.enqueue_ops.append(gpucopy_op)
        train_and_update_ops = tf.group(*([train_op] + update_ops))
        all_training_ops = (self.enqueue_ops + [train_and_update_ops])
        return total_loss_avg, self.learning_rate, all_training_ops
    def init(self, sess):
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
    def sync(self, sess):
        sync_op = hvd.broadcast_global_variables(0)
        sess.run(sync_op)
    def prefill_pipeline(self, sess):
        # Pre-fill the input pipeline with data
        for i in range(len(self.enqueue_ops)):
            sess.run(self.enqueue_ops[:i+1])