# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""ImageNet preprocessing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow.compat.v1 as tf


def _resize_image(image, image_size, method=None):
  if method is not None:
    return tf.image.resize([image], [image_size, image_size], method)[0]
  return tf.image.resize_bicubic([image], [image_size, image_size])[0]

def _decode_and_center_crop(image, image_size, resize_method=None, central_fraction=0.875):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  minimum = tf.minimum(image_height, image_width)
  padded_center_crop_size = tf.cast(
      (central_fraction *
       tf.cast(minimum, tf.float32)),
       tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  image = tf.image.crop_to_bounding_box(image,
                                        offset_height, offset_width,
                                        padded_center_crop_size,
                                        padded_center_crop_size)

  image = _resize_image(image, image_size, resize_method)
  return image


def preprocess_image(image,
                     image_size,
                     resize_method=None):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image
    image_size: image size.
    resize_method: if None, use bicubic.

  Returns:
    A preprocessed image `Tensor`.
  """

  image = _decode_and_center_crop(image, image_size, resize_method)
  image = tf.reshape(image, [image_size, image_size, 3])
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = image - 127.
  image = image / 128.
  return image



