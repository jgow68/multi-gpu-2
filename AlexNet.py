def inference_alexnet_owt(net, input_layer):
    """Alexnet One Weird Trick model
    https://arxiv.org/abs/1404.5997
    """
    net.use_batch_norm = False
    x = net.input_layer(input_layer)
    # Note: VALID requires padding the images by 3 in width and height
    x = net.conv(x, 64, (11,11), (4,4), 'VALID')
    x = net.pool(x, 'MAX', (3,3))
    x = net.conv(x, 192,   (5,5))
    x = net.pool(x, 'MAX', (3,3))
    x = net.conv(x, 384,   (3,3))
    x = net.conv(x, 256,   (3,3))
    x = net.conv(x, 256,   (3,3))
    x = net.pool(x, 'MAX', (3,3))
    x = net.flatten(x)
    x = net.fully_connected(x, 4096)
    x = net.dropout(x)
    x = net.fully_connected(x, 4096)
    x = net.dropout(x)
    return x