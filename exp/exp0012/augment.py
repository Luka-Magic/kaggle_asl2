import tensorflow as tf
import numpy as np


def interp1d_(x, target_len, method='random'):
    target_len = tf.maximum(1, target_len)
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bicubic')
            else:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'nearest')
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
    return x


def resample(x, rate=(0.8, 1.2)):
    # shape: [frame_len, 543, 3]
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate*tf.cast(length, tf.float32), tf.int32)
    new_x = interp1d_(x, new_size)
    return new_x


def temporal_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    # shape: [frame_len, n_landmarks, 3]
    l = tf.shape(x)[0]
    n_landmarks = tf.shape(x)[1]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform(
        (), 0, tf.clip_by_value(l-mask_size, 1, l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(x, tf.range(
        mask_offset, mask_offset+mask_size)[..., None], tf.fill([mask_size, n_landmarks, 3], mask_value))
    return x


def drop_random_frames(x, size=(0.05, 0.1), mask_value=float('nan')):
    # drop random frames
    l = tf.shape(x)[0]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    # choice random mask_size frames
    mask_random_frames = np.random.choice(
        int(l), int(l - mask_size), replace=False)
    x = tf.gather(x, mask_random_frames, axis=0)
    return x


def spatial_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    # shape: [frame_len, n_landmarks, 3]
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x < x[..., 0]) & (
        x[..., 0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y < x[..., 1]) & (
        x[..., 1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[..., None], mask_value, x)
    return x


def spatial_random_affine(xyz,
                          scale=(0.8, 1.2),
                          shear=(-0.15, 0.15),
                          shift=(-0.1, 0.1),
                          degree=(-30, 30),
                          ):
    center = tf.constant([0.5, 0.5])
    if scale is not None:
        scale = tf.random.uniform((), *scale)
        xyz = scale*xyz

    if shear is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = tf.identity([
            [1., shear_x],
            [shear_y, 1.]
        ])
        xy = xy @ shear_mat
        center = center + [shear_y, shear_x]
        xyz = tf.concat([xy, z], axis=-1)

    if degree is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        xy -= center
        degree = tf.random.uniform((), *degree)
        radian = degree/180*np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity([
            [c, s],
            [-s, c],
        ])
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = tf.concat([xy, z], axis=-1)

    if shift is not None:
        shift = tf.random.uniform((), *shift)
        xyz = xyz + shift

    return xyz


def augment_fn(x, always=False, max_len=None):
    if tf.random.uniform(()) < 0.4 or always:  # 40%
        x = resample(x, (0.85, 1.15))
    elif tf.random.uniform(()) < 0.5 or always:  # 30%
        x = drop_random_frames(x, (0.05, 0.1))

    if tf.random.uniform(()) < 0.75 or always:
        x = spatial_random_affine(
            x,
            scale=(0.9, 1.1),
            shear=(-0.1, 0.1),
            shift=(-0.1, 0.1),
            degree=(-5, 5),
        )
    if tf.random.uniform(()) < 0.5 or always:
        x = spatial_mask(x, (0.05, 0.1))
    return x
