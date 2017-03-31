import mxnet as mx
from common import conv_act_layer
from common import multibox_layer


def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    """
        TODO : replace by conv_act_layer
    """
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act


def Inception7A(data,
                num_1x1,
                num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5,
                pool, proj,
                name):
    tower_1x1 = Conv(data, num_1x1, name=('%s_conv' % name))
    tower_5x5 = Conv(data, num_5x5_red, name=('%s_tower' % name), suffix='_conv')
    tower_5x5 = Conv(tower_5x5, num_5x5, kernel=(5, 5), pad=(2, 2), name=('%s_tower' % name), suffix='_conv_1')
    tower_3x3 = Conv(data, num_3x3_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_3x3 = Conv(tower_3x3, num_3x3_1, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3 = Conv(tower_3x3, num_3x3_2, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_2')
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(pooling, proj, name=('%s_tower_2' %  name), suffix='_conv')
    concat = mx.sym.Concat(*[tower_1x1, tower_5x5, tower_3x3, cproj], name='ch_concat_%s_chconcat' % name)
    return concat


# First Downsample
def Inception7B(data,
                num_3x3,
                num_d3x3_red, num_d3x3_1, num_d3x3_2,
                pool,
                name):
    tower_3x3 = Conv(data, num_3x3, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=('%s_conv' % name))
    tower_d3x3 = Conv(data, num_d3x3_red, name=('%s_tower' % name), suffix='_conv')
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name=('%s_tower' % name), suffix='_conv_1')
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_2, kernel=(3, 3), pad=(0, 0), stride=(2, 2), name=('%s_tower' % name), suffix='_conv_2')
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0,0), pool_type="max", name=('max_pool_%s_pool' % name))
    concat = mx.sym.Concat(*[tower_3x3, tower_d3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


def Inception7C(data,
                num_1x1,
                num_d7_red, num_d7_1, num_d7_2,
                num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4,
                pool, proj,
                name):
    tower_1x1 = Conv(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_conv' % name))
    tower_d7 = Conv(data=data, num_filter=num_d7_red, name=('%s_tower' % name), suffix='_conv')
    tower_d7 = Conv(data=tower_d7, num_filter=num_d7_1, kernel=(1, 7), pad=(0, 3), name=('%s_tower' % name), suffix='_conv_1')
    tower_d7 = Conv(data=tower_d7, num_filter=num_d7_2, kernel=(7, 1), pad=(3, 0), name=('%s_tower' % name), suffix='_conv_2')
    tower_q7 = Conv(data=data, num_filter=num_q7_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_q7 = Conv(data=tower_q7, num_filter=num_q7_1, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_q7 = Conv(data=tower_q7, num_filter=num_q7_2, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_2')
    tower_q7 = Conv(data=tower_q7, num_filter=num_q7_3, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_3')
    tower_q7 = Conv(data=tower_q7, num_filter=num_q7_4, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_4')
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_tower_2' %  name), suffix='_conv')
    # concat
    concat = mx.sym.Concat(*[tower_1x1, tower_d7, tower_q7, cproj], name='ch_concat_%s_chconcat' % name)
    return concat


def Inception7D(data,
                num_3x3_red, num_3x3,
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3,
                pool,
                name):
    tower_3x3 = Conv(data=data, num_filter=num_3x3_red, name=('%s_tower' % name), suffix='_conv')
    tower_3x3 = Conv(data=tower_3x3, num_filter=num_3x3, kernel=(3, 3), pad=(0,0), stride=(2, 2), name=('%s_tower' % name), suffix='_conv_1')
    tower_d7_3x3 = Conv(data=data, num_filter=num_d7_3x3_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=num_d7_1, kernel=(1, 7), pad=(0, 3), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=num_d7_2, kernel=(7, 1), pad=(3, 0), name=('%s_tower_1' % name), suffix='_conv_2')
    tower_d7_3x3 = Conv(data=tower_d7_3x3, num_filter=num_d7_3x3, kernel=(3, 3), stride=(2, 2), name=('%s_tower_1' % name), suffix='_conv_3')
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    # concat
    concat = mx.sym.Concat(*[tower_3x3, tower_d7_3x3, pooling], name='ch_concat_%s_chconcat' % name)
    return concat


def Inception7E(data,
                num_1x1,
                num_d3_red, num_d3_1, num_d3_2,
                num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2,
                pool, proj,
                name):
    tower_1x1 = Conv(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_conv' % name))
    tower_d3 = Conv(data=data, num_filter=num_d3_red, name=('%s_tower' % name), suffix='_conv')
    tower_d3_a = Conv(data=tower_d3, num_filter=num_d3_1, kernel=(1, 3), pad=(0, 1), name=('%s_tower' % name), suffix='_mixed_conv')
    tower_d3_b = Conv(data=tower_d3, num_filter=num_d3_2, kernel=(3, 1), pad=(1, 0), name=('%s_tower' % name), suffix='_mixed_conv_1')
    tower_3x3_d3 = Conv(data=data, num_filter=num_3x3_d3_red, name=('%s_tower_1' % name), suffix='_conv')
    tower_3x3_d3 = Conv(data=tower_3x3_d3, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name), suffix='_conv_1')
    tower_3x3_d3_a = Conv(data=tower_3x3_d3, num_filter=num_3x3_d3_1, kernel=(1, 3), pad=(0, 1), name=('%s_tower_1' % name), suffix='_mixed_conv')
    tower_3x3_d3_b = Conv(data=tower_3x3_d3, num_filter=num_3x3_d3_2, kernel=(3, 1), pad=(1, 0), name=('%s_tower_1' % name), suffix='_mixed_conv_1')
    pooling = mx.sym.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_tower_2' %  name), suffix='_conv')
    # concat
    concat = mx.sym.Concat(*[tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj], name='ch_concat_%s_chconcat' % name)
    return concat


def get_symbol_train(num_classes=20):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is a training network with losses

    Inception V3, suitable for images with around 299 x 299
    Reference:
    Szegedy, Christian, et al. "Rethinking the Inception Architecture for Computer Vision." arXiv preprint arXiv:1512.00567 (2015).

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background

    Returns:
    ----------
    mx.Symbol
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    print("===============================")
    print("get symbol train : inception v3")
    print("===============================")
    # ====== inception v3 layer ======
    # stage 1
    conv = Conv(data, 32, kernel=(3, 3), stride=(2, 2), name="conv")
    conv_1 = Conv(conv, 32, kernel=(3, 3), name="conv_1")
    conv_2 = Conv(conv_1, 64, kernel=(3, 3), pad=(1, 1), name="conv_2")
    pool = mx.sym.Pooling(data=conv_2, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool")
    # stage 2
    conv_3 = Conv(pool, 80, kernel=(1, 1), name="conv_3")
    conv_4 = Conv(conv_3, 192, kernel=(3, 3), name="conv_4")
    pool1 = mx.sym.Pooling(data=conv_4, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool1")
    # stage 3
    in3a = Inception7A(pool1, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 32, "mixed")
    in3b = Inception7A(in3a, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64, "mixed_1")
    in3c = Inception7A(in3b, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64, "mixed_2")
    in3d = Inception7B(in3c, 384,
                       64, 96, 96,
                       "max", "mixed_3")
    # stage 4
    in4a = Inception7C(in3d, 192,
                       128, 128, 192,
                       128, 128, 128, 128, 192,
                       "avg", 192, "mixed_4")
    in4b = Inception7C(in4a, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192, "mixed_5")
    in4c = Inception7C(in4b, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192, "mixed_6")
    in4d = Inception7C(in4c, 192,
                       192, 192, 192,
                       192, 192, 192, 192, 192,
                       "avg", 192, "mixed_7")
    in4e = Inception7D(in4d, 192, 320,
                       192, 192, 192, 192,
                       "max", "mixed_8")
    # stage 5
    in5a = Inception7E(in4e, 320,
                       384, 384, 384,
                       448, 384, 384, 384,
                       "avg", 192, "mixed_9")
    in5b = Inception7E(in5a, 320,
                       384, 384, 384,
                       448, 384, 384, 384,
                       "max", 192, "mixed_10")

    # group 6
    conv6 = mx.symbol.Convolution(
        data=in5b, kernel=(3, 3), pad=(6, 6), dilate=(6, 6),
        num_filter=1024, name="conv6")
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="conv7")
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    # ====== ssd extra layers ======
    conv8_1, relu8_1 = conv_act_layer(drop7, "8_1", 256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type="relu", use_batchnorm=False)
    conv8_2, relu8_2 = conv_act_layer(relu8_1, "8_2", 512, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type="relu", use_batchnorm=False)
    conv9_1, relu9_1 = conv_act_layer(relu8_2, "9_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type="relu", use_batchnorm=False)
    conv9_2, relu9_2 = conv_act_layer(relu9_1, "9_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type="relu", use_batchnorm=False)
    conv10_1, relu10_1 = conv_act_layer(relu9_2, "10_1", 128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type="relu", use_batchnorm=False)
    conv10_2, relu10_2 = conv_act_layer(relu10_1, "10_2", 256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), act_type="relu", use_batchnorm=False)
    # global Pooling
    pool10 = mx.symbol.Pooling(data=relu10_2, pool_type="avg", global_pool=True, kernel=(1, 1), name='pool10')

    # TODO specific parameters for inception network we replace relu4_3, relu7 by in4a, in5a don't find good equivalent layer in inception yet
    from_layers = [in3d, in4e, relu7, relu8_2, relu9_2, relu10_2, pool10]  # relu4_3, relu7,
    sizes = [[.1], [.2, .276], [.2, .321], [.38, .461], [.56, .644], [.74, .825], [.92, 1.01]]  # [.1], [.2, .276], 
    ratios = [[1, 2, .5], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3], [1, 2, .5, 3, 1./3]]  # [1, 2, .5], [1, 2, .5, 3, 1./3], 
    normalizations = [20, 20, -1, -1, -1, -1, -1]  # [20, -1, -1, -1, -1, -1]
    num_channels = [768, 1280]

    loc_preds, cls_preds, anchor_boxes = multibox_layer(
        from_layers, num_classes, sizes=sizes, ratios=ratios,
        normalization=normalizations, num_channels=num_channels, clip=True,
        interm_layer=0)

    tmp = mx.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5,
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0,
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(
        data=cls_preds, label=cls_target, ignore_label=-1, use_ignore=True,
        grad_scale=3., multi_output=True, normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(
        name="loc_loss_", data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label])
    return out


def get_symbol(num_classes=20, nms_thresh=0.5, force_suppress=True):
    """
    Single-shot multi-box detection with VGG 16 layers ConvNet
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    This is the detection network

    Parameters:
    ----------
    num_classes: int
        number of object classes not including background
    nms_thresh : float
        threshold of overlap for non-maximum suppression

    Returns:
    ----------
    mx.Symbol
    """
    net = get_symbol_train(num_classes)
    #print net.get_internals().list_outputs()
    cls_preds = net.get_internals()["multibox_cls_pred_output"]
    loc_preds = net.get_internals()["multibox_loc_pred_output"]
    anchor_boxes = net.get_internals()["multibox_anchors_output"]

    cls_prob = mx.symbol.SoftmaxActivation(data=cls_preds, mode='channel', name='cls_prob')
    # group output
    # out = mx.symbol.Group([loc_preds, cls_preds, anchor_boxes])
    out = mx.symbol.MultiBoxDetection(
        *[cls_prob, loc_preds, anchor_boxes], name="detection",
        nms_threshold=nms_thresh, force_suppress=force_suppress, variances=(0.1, 0.1, 0.2, 0.2))
    return out
