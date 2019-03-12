import os

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

import settings

class ImagePreprocessor(object):
    def __init__(self, height, width, subset='train', dtype=tf.uint8):
        self.height = height
        self.width  = width
        self.subset = subset
        self.dtype = dtype
        self.nsummary = 10 # Max no. images to generate summaries for
        
    def __deserialize_image_record(self, record):
        feature_map = {
            'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
            'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
            'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
        }
        with tf.name_scope('deserialize_image_record'):
            obj = tf.parse_single_example(record, feature_map)
            imgdata = obj['image/encoded']
            label   = tf.cast(obj['image/class/label'], tf.int32)
            bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                                for x in ['ymin', 'xmin', 'ymax', 'xmax']])
            bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
            text    = obj['image/class/text']
            return imgdata, label, bbox, text

    def __decode_jpeg(self, imgdata, channels=3):
        return tf.image.decode_jpeg(imgdata, channels=channels,
                                    fancy_upscaling=False,
                                    dct_method='INTEGER_FAST')

    def __decode_png(self, imgdata, channels=3):
        return tf.image.decode_png(imgdata, channels=channels)

    def __distort_image_color(self, image, order):
        with tf.name_scope('distort_color'):
            image /= 255.
            brightness = lambda img: tf.image.random_brightness(img, max_delta=32. / 255.)
            saturation = lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5)
            hue        = lambda img: tf.image.random_hue(img, max_delta=0.2)
            contrast   = lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5)
            if order == 0: ops = [brightness, saturation, hue, contrast]
            else:          ops = [brightness, contrast, saturation, hue]
            for op in ops:
                image = op(image)
            # The random_* ops do not necessarily clamp the output range
            image = tf.clip_by_value(image, 0.0, 1.0)
            # Restore the original scaling
            image *= 255
            return image

    def __random_crop_and_resize_image(self, image, bbox, height, width):
        with tf.name_scope('random_crop_and_resize'):

            bbox_begin, bbox_size, distorted_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=0.1,
                aspect_ratio_range=[0.8, 1.25],
                area_range=[0.1, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            # Crop the image to the distorted bounding box
            image = tf.slice(image, bbox_begin, bbox_size)
            
            # Resize to the desired output size
            image = tf.image.resize_images(
                image,
                [height, width],
                tf.image.ResizeMethod.BILINEAR,
                align_corners=False)
            image.set_shape([height, width, 3])
            return image        
        
    def preprocess(self, imgdata, bbox, thread_id):
        with tf.name_scope('preprocess_image'):
            try:
                image = self.__decode_jpeg(imgdata)
            except:
                image = self.__decode_png(imgdata)
            if thread_id < self.nsummary:
                image_with_bbox = tf.image.draw_bounding_boxes(
                    tf.expand_dims(tf.to_float(image), 0), bbox)
                tf.summary.image('original_image_and_bbox', image_with_bbox)
            image = self.__random_crop_and_resize_image(image, bbox,
                                                 self.height, self.width)
            if thread_id < self.nsummary:
                tf.summary.image('cropped_resized_image',
                                 tf.expand_dims(image, 0))
            image = tf.image.random_flip_left_right(image)
            if thread_id < self.nsummary:
                tf.summary.image('flipped_image',
                                 tf.expand_dims(image, 0))
            if settings.FLAGS.distort_color:
                image = self.__distort_image_color(image, order=thread_id%2)
                if thread_id < self.nsummary:
                    tf.summary.image('distorted_color_image',
                                     tf.expand_dims(image, 0))
        return image
    
    def minibatch(self, batch_size):
        record_input = data_flow_ops.RecordInput(
            file_pattern=os.path.join(settings.FLAGS.data_dir, '%s-*' % self.subset),
            parallelism=64,
            # Note: This causes deadlock during init if larger than dataset
            buffer_size=settings.FLAGS.input_buffer_size,
            batch_size=batch_size)
        records = record_input.get_yield_op()
        # Split batch into individual images
        records = tf.split(records, batch_size, 0)
        records = [tf.reshape(record, []) for record in records]
        # Deserialize and preprocess images into batches for each device
        images = []
        labels = []
        with tf.name_scope('input_pipeline'):
            for i, record in enumerate(records):
                imgdata, label, bbox, text = self.__deserialize_image_record(record)
                image = self.preprocess(imgdata, bbox, thread_id=i)
                label -= 1 # Change to 0-based (don't use background class)
                images.append(image)
                labels.append(label)
            # Stack images back into a single tensor
            images = tf.parallel_stack(images)
            labels = tf.concat(labels, 0)
            images = tf.reshape(images, [-1, self.height, self.width, 3])
            images = tf.clip_by_value(images, 0., 255.)
            images = tf.cast(images, self.dtype)
        return images, labels
