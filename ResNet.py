import tensorflow as tf

def resnet_bottleneck_v1(net, input_layer, depth, depth_bottleneck, stride,
                         basic=False):
    num_inputs = input_layer.get_shape().as_list()[1]
    x = input_layer
    s = stride
    with tf.name_scope('resnet_v1'):
        if depth == num_inputs:
            if stride == 1:
                shortcut = input_layer
            else:
                shortcut = net.pool(x, 'MAX', (1,1), (s,s))
        else:
            shortcut = net.conv(x, depth, (1,1), (s,s), activation='LINEAR')
        if basic:
            x = net.conv(x, depth_bottleneck, (3,3), (s,s), padding='SAME_RESNET')
            x = net.conv(x, depth,            (3,3), activation='LINEAR')
        else:
            x = net.conv(x, depth_bottleneck, (1,1), (s,s))
            x = net.conv(x, depth_bottleneck, (3,3), padding='SAME')
            x = net.conv(x, depth,            (1,1), activation='LINEAR')
        x = net.activate(x + shortcut)
        return x
    


def inference_residual(net, input_layer, layer_counts, bottleneck_callback):
    net.use_batch_norm = True
    x = net.input_layer(input_layer)
    x = net.conv(x, 64,    (7,7), (2,2), padding='SAME_RESNET')
    x = net.pool(x, 'MAX', (3,3), (2,2), padding='SAME')
    for i in range(layer_counts[0]):
        x = bottleneck_callback(net, x,  256,  64, 1)
    for i in range(layer_counts[1]):
        x = bottleneck_callback(net, x, 512, 128, 2 if i==0 else 1)
    for i in range(layer_counts[2]):
        x = bottleneck_callback(net, x, 1024, 256, 2 if i==0 else 1)
    for i in range(layer_counts[3]):
        x = bottleneck_callback(net, x, 2048, 512, 2 if i==0 else 1)
    x = net.spatial_avg(x)
    return x

def inference_resnet_v1_basic_impl(net, input_layer, layer_counts):
    basic_resnet_bottleneck_callback = partial(resnet_bottleneck_v1, basic=True)
    return inference_residual(net, input_layer, layer_counts, basic_resnet_bottleneck_callback)

def inference_resnet_v1_impl(net, input_layer, layer_counts):
    return inference_residual(net, input_layer, layer_counts, resnet_bottleneck_v1)

def inference_resnet_v1(net, input_layer, nlayer):
    """Deep Residual Networks family of models
    https://arxiv.org/abs/1512.03385
    """
    if   nlayer ==  18: return inference_resnet_v1_basic_impl(net, input_layer, [2,2, 2,2])
    elif nlayer ==  34: return inference_resnet_v1_basic_impl(net, input_layer, [3,4, 6,3])
    elif nlayer ==  50: return inference_resnet_v1_impl(net, input_layer, [3,4, 6,3])
    elif nlayer == 101: return inference_resnet_v1_impl(net, input_layer, [3,4,23,3])
    elif nlayer == 152: return inference_resnet_v1_impl(net, input_layer, [3,8,36,3])
    else: raise ValueError("Invalid nlayer (%i); must be one of: 18,34,50,101,152" %
                           nlayer)

