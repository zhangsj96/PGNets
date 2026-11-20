"""Grad-CAM: which parts of the image drive the network's answer."""

from __future__ import annotations

import numpy as np


def last_conv_layer(model):
    """Last spatial (4-D) feature map before the pooling/flatten head."""
    for layer in reversed(model.layers):
        if "pool" in layer.name or "flatten" in layer.name:
            continue
        if len(layer.output.shape) == 4:
            return layer
    raise ValueError("no spatial layer found")


def gradcam(model, x64, output_index=None, layer_name=None):
    """Heat map (64x64, in [0, 1]) for ``output_index`` (default: the top output).

    For the regression network use ``output_index=0`` (planet mass), as in the paper.
    """
    import tensorflow as tf

    layer = model.get_layer(layer_name) if layer_name else last_conv_layer(model)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    img = tf.convert_to_tensor(np.asarray(x64, "float32")[None, :, :, None])
    with tf.GradientTape() as tape:
        conv, preds = grad_model(img)
        idx = int(tf.argmax(preds[0])) if output_index is None else output_index
        target = preds[:, idx]
    grads = tape.gradient(target, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    if cam.max() > 0:
        cam /= cam.max()
    from scipy.ndimage import zoom

    return np.clip(zoom(cam, 64 / cam.shape[0], order=1), 0, 1)
