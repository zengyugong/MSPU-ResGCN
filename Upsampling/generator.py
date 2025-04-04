# -*- coding: utf-8 -*-
# @Description :
# @Author      : Guocheng Qian
# @Email       : guocheng.qian@kaust.edu.sa

import tensorflow as tf
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from Common.pointnet_util import pointnet_sa_module, pointnet_fp_module

class PUGCN_Residual(object):
    """
    Improved PU-GCN with Residual Connections for Enhanced Learning
    """

    def __init__(self, opts, is_training, name="Generator_Residual"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            
            # Feature Extraction with Residual Connections
            features, idx = ops.feature_extractor_updated(inputs,
                                                  self.opts.block, self.opts.n_blocks,
                                                  self.opts.channels, self.opts.k, self.opts.d,
                                                  use_global_pooling=self.opts.use_global_pooling,
                                                  scope='feature_extraction', is_training=self.is_training,
                                                  bn_decay=None)
            
            print(features.shape)

            # # # Residual Connection 1
            # features_residual = ops.conv2d(features, 193, [1, 1], scope='residual_1', 
            #                                activation_fn=tf.nn.relu, is_training=self.is_training)
            # features += features_residual

            # # Residual Connection 1 - Ensure Same Feature Dimension
            # features_residual = ops.conv2d(features, features.shape[-1], [1, 1], scope='residual_1',
            #                             activation_fn=tf.nn.relu, is_training=self.is_training)
            # # Element-wise addition with correctly aligned dimensions
            # features += features_residual
            
            # # Residual Connection 2
            # features_residual = ops.conv2d(features, features.shape[-1], [1, 1], scope='residual_2',
            #                                activation_fn=tf.nn.relu, is_training=self.is_training)
            # features += features_residual

            # features_residual = ops.conv2d(features, features.shape[-1], [1, 1], scope='residual_3',
            #                                activation_fn=tf.nn.relu, is_training=self.is_training)
            # features += features_residual

            # Upsampling Block with Residual Connections
            H = ops.up_unit(features, self.up_ratio_real,
                            self.opts.upsampler,
                            k=self.opts.k,
                            idx=idx,
                            scope="up_block",
                            use_att=self.opts.use_att,
                            is_training=self.is_training, bn_decay=None)
            
            print(H.shape)

            # # Residual Connection 1
            # H_residual = ops.conv2d(H, H.shape[-1], [1, 1], scope='residual_2',
            #                         activation_fn=tf.nn.relu, is_training=self.is_training)
            # H += H_residual

            # # Residual Connection 2
            # H_residual = ops.conv2d(H, H.shape[-1], [1, 1], scope='residual_3',
            #                         activation_fn=tf.nn.relu, is_training=self.is_training)
            # H += H_residual
            
            # Residual Connection 1
            H_residual1 = ops.conv2d(H, H.shape[-1], [1, 1], scope='residual_2',
                                    activation_fn=tf.nn.relu, is_training=self.is_training)
            

            # Residual Connection 2
            H_residual2 = ops.conv2d(H, H.shape[-1], [3, 3], scope='residual_3',
                                    activation_fn=tf.nn.relu, is_training=self.is_training)
            
            # Residual Connection 4
            H_residual3 = ops.conv2d(H, H.shape[-1], [5, 5], scope='residual_4',
                                    activation_fn=tf.nn.relu, is_training=self.is_training)
            

            
            H_residual = tf.concat([H_residual1, H_residual2, H_residual3], axis=-1)
            # H_residual = tf.concat([H_residual1, H_residual2, H_residual3, H_residual4], axis=-1)

            # Projection to Match the Original Shape of H
            H_residual = ops.conv2d(H_residual, H.shape[-1], [1, 1], scope='residual_projection',
                         activation_fn=None, is_training=self.is_training)

            H += H_residual

            # Try layer normalization (not good)
            # H_residual = ops.conv2d(H, 64, [1, 1], scope='residual_2',
            #             activation_fn=None, is_training=self.is_training)
            # H_residual = tf.contrib.layers.layer_norm(H_residual,scope='layer_norm')
            # H_residual = tf.nn.relu(H_residual)
            # H += H_residual
            
            # Use gate residual function (good but parameter larger 121k)
            # gate = tf.sigmoid(ops.conv2d(H, 64, [1, 1], scope='gate'))  # Gate values range [0, 1]

            # # Residual Branch (NO ReLU here)
            # H_residual = ops.conv2d(H, 64, [1, 1], scope='residual_2', 
            #                         activation_fn=None, is_training=self.is_training)

            # # Combine Original + Residual via Gate
            # H = gate * H + (1 - gate) * H_residual
            # H += H_residual

            # Attention mechanism
            # attention_weights = tf.nn.softmax(ops.conv2d(H_residual, 1, [1, 1], scope='attention',
            #                                      activation_fn=None, is_training=self.is_training), axis=1)
            # H = H * attention_weights

            # Dropout for Regularization
            # H = tf.layers.dropout(H, rate=0.1, training=self.is_training)

            # Final Coordinate Prediction
            coord = ops.conv2d(H, 32, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            
            outputs = tf.squeeze(coord, [2])

            # Residual Connection for Coordinate Outputs
            if self.up_ratio_real > self.up_ratio:
                outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
            outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
                                  [inputs.shape[0], self.num_point * self.up_ratio, -1])

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs

# class PUGCN_Residual_Dynamic(object):
#     """
#     Improved PU-GCN with Residual Connections for Enhanced Learning
#     """

#     def __init__(self, opts, is_training, name="Generator_Residual"):
#         self.opts = opts
#         self.is_training = is_training
#         self.name = name
#         self.reuse = False
#         self.num_point = self.opts.patch_num_point
#         self.up_ratio = self.opts.up_ratio
#         self.up_ratio_real = self.up_ratio + self.opts.more_up
#         self.out_num_point = int(self.num_point * self.up_ratio)

#     def dynamic_graph(self, inputs, k=20, scope='dynamic_graph'):
#         with tf.variable_scope(scope):
#             # Reshape inputs to [B, N, C] if needed
#             inputs_reshaped = tf.squeeze(inputs, [2]) # remove the 3rd dimension.
#             pairwise_dist = ops.pairwise_distance(inputs_reshaped)
#             #idx = ops.knn(pairwise_dist, k=k)
#             edge_features,idx = ops.get_edge_feature(inputs_reshaped)
#             edge_features = tf.expand_dims(edge_features, axis=2) # add back the 3rd dimension.
#             return edge_features

#     def __call__(self, inputs):
#         with tf.variable_scope(self.name, reuse=self.reuse):
            
#             # Feature Extraction with Residual Connections
#             features, idx = ops.feature_extractor(inputs,
#                                                   self.opts.block, self.opts.n_blocks,
#                                                   self.opts.channels, self.opts.k, self.opts.d,
#                                                   use_global_pooling=self.opts.use_global_pooling,
#                                                   scope='feature_extraction', is_training=self.is_training,
#                                                   bn_decay=None)

#             # # Residual Connection 1
#             # features_residual = ops.conv2d(features, 64, [1, 1], scope='residual_1', 
#             #                                activation_fn=tf.nn.relu, is_training=self.is_training)
#             # features += features_residual

#             # Dynamic Graph Feature Enhancement
#             dg_features = self.dynamic_graph(features, k=self.opts.k)
#             features = tf.concat([features, dg_features], axis=-1)

#             # Residual Connection 1 - Ensure Same Feature Dimension
#             features_residual = ops.conv2d(features, 193, [1, 1], scope='residual_1',
#                                         activation_fn=tf.nn.relu, is_training=self.is_training)

#             # Element-wise addition with correctly aligned dimensions
#             features += features_residual

#             # Upsampling Block with Residual Connections
#             H = ops.up_unit(features, self.up_ratio_real,
#                             self.opts.upsampler,
#                             k=self.opts.k,
#                             idx=idx,
#                             scope="up_block",
#                             use_att=self.opts.use_att,
#                             is_training=self.is_training, bn_decay=None)

#             # Residual Connection 2
#             H_residual = ops.conv2d(H, 64, [1, 1], scope='residual_2',
#                                     activation_fn=tf.nn.relu, is_training=self.is_training)

#             # Attention mechanism
#             attention_weights = tf.nn.softmax(ops.conv2d(H_residual, 1, [1, 1], scope='attention',
#                                                  activation_fn=None, is_training=self.is_training), axis=1)
#             H = H * attention_weights

#             # Dropout for Regularization
#             # H = tf.layers.dropout(H, rate=0.3, training=self.is_training)

#             # Final Coordinate Prediction
#             coord = ops.conv2d(H, 32, [1, 1],
#                                padding='VALID', stride=[1, 1],
#                                bn=False, is_training=self.is_training,
#                                scope='fc_layer1', bn_decay=None,
#                                activation_fn=tf.nn.leaky_relu
#                                )

#             coord = ops.conv2d(coord, 3, [1, 1],
#                                padding='VALID', stride=[1, 1],
#                                bn=False, is_training=self.is_training,
#                                scope='fc_layer2', bn_decay=None,
#                                activation_fn=None, weight_decay=0.0)
            
#             outputs = tf.squeeze(coord, [2])

#             # Residual Connection for Coordinate Outputs
#             if self.up_ratio_real > self.up_ratio:
#                 outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
#             outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
#                                   [inputs.shape[0], self.num_point * self.up_ratio, -1])

#         self.reuse = True
#         self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
#         return outputs

# class PUGCN_Res_MHA_Dynamic(object):
#     def __init__(self, opts, is_training, name="Generator_Residual"):
#         self.opts = opts
#         self.is_training = is_training
#         self.name = name
#         self.reuse = False
#         self.num_point = self.opts.patch_num_point
#         self.up_ratio = self.opts.up_ratio
#         self.up_ratio_real = self.up_ratio + self.opts.more_up
#         self.out_num_point = int(self.num_point * self.up_ratio)

#     def multi_head_attention(self, H, num_heads=4, key_dim=64):
#         batch_size = tf.shape(H)[0]

#         # Linear projections for Q, K, V
#         Q = ops.conv2d(H, num_heads * key_dim, [1, 1], scope='Q')
#         K = ops.conv2d(H, num_heads * key_dim, [1, 1], scope='K')
#         V = ops.conv2d(H, num_heads * key_dim, [1, 1], scope='V')

#         # Reshape to [batch_size, num_points, num_heads, key_dim]
#         Q_ = tf.reshape(Q, [batch_size, -1, num_heads, key_dim])
#         K_ = tf.reshape(K, [batch_size, -1, num_heads, key_dim])
#         V_ = tf.reshape(V, [batch_size, -1, num_heads, key_dim])

#         # Transpose to [batch_size, num_heads, num_points, key_dim] for matmul
#         Q_ = tf.transpose(Q_, [0, 2, 1, 3])
#         K_ = tf.transpose(K_, [0, 2, 1, 3])
#         V_ = tf.transpose(V_, [0, 2, 1, 3])

#         assert Q_.shape == K_.shape, "Mismatch in Q and K dimensions."
#         assert K_.shape == V_.shape, "Mismatch in K and V dimensions."

#         # Scaled Dot-Product Attention
#         scaled_dot_product = tf.matmul(Q_, K_, transpose_b=True) / tf.sqrt(tf.cast(key_dim, tf.float32))
#         attention_weights = tf.nn.softmax(scaled_dot_product, axis=-1)

#         # Attention output
#         output = tf.matmul(attention_weights, V_)

#         # Reshape back to original shape [batch_size, num_points, feature_dim]
#         output = tf.reshape(output, [batch_size, -1, num_heads * key_dim])
#         return output

#     def __call__(self, inputs):
#         with tf.variable_scope(self.name, reuse=self.reuse):
#             features, idx = ops.feature_extractor(inputs, self.opts.block, self.opts.n_blocks,
#                                                   self.opts.channels, self.opts.k, self.opts.d,
#                                                   use_global_pooling=self.opts.use_global_pooling,
#                                                   scope='feature_extraction',
#                                                   is_training=self.is_training, bn_decay=None)

#             features_residual = ops.conv2d(features, 193, [1, 1], scope='residual_1',
#                                            activation_fn=tf.nn.relu, is_training=self.is_training)
#             features += features_residual

#             H = ops.up_unit(features, self.up_ratio_real, self.opts.upsampler, k=self.opts.k,
#                             idx=idx, scope="up_block", use_att=self.opts.use_att,
#                             is_training=self.is_training, bn_decay=None)

#             H = self.multi_head_attention(H)

#             H = tf.expand_dims(H, axis=1)  # Adds a new axis at position 1: [64, 1, 1024, 256]

#             coord = ops.conv2d(H, 32, [1, 1], padding='VALID', stride=[1, 1],
#                                bn=False, is_training=self.is_training,
#                                scope='fc_layer1', bn_decay=None,
#                                activation_fn=tf.nn.leaky_relu)

#             coord = ops.conv2d(coord, 3, [1, 1], padding='VALID', stride=[1, 1],
#                                bn=False, is_training=self.is_training,
#                                scope='fc_layer2', bn_decay=None,
#                                activation_fn=None, weight_decay=0.0)

#             #outputs = tf.squeeze(coord, [2])
#             outputs = tf.squeeze(coord, [1])

#             if self.up_ratio_real > self.up_ratio:
#                 outputs = ops.gather_point(outputs, ops.farthest_point_sample(self.out_num_point, outputs))
#             outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
#                                   [inputs.shape[0], self.num_point * self.up_ratio, -1])

#         self.reuse = True
#         self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
#         return outputs
    

class PUGAN(object):
    """
    [PU-GAN](https://arxiv.org/abs/1907.10844)
    """
    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        channels = 24  # 24 for PU-GAN
        with tf.variable_scope(self.name, reuse=self.reuse):
            features = ops.feature_extraction(inputs,
                                              channels,
                                              scope='feature_extraction', is_training=self.is_training,
                                              bn_decay=None)
            H = ops.up_projection_unit(features, self.up_ratio_real,
                                       scope="up_projection_unit",
                                       is_training=self.is_training, bn_decay=None)
            coord = ops.conv2d(H, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None)

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])

            if self.up_ratio_real > self.up_ratio:
                outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs




class MPU(object):
    """
    [MPU (3PU)](https://arxiv.org/abs/1811.11286)
    """
    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        channels = 12   # 12 for MPU
        with tf.variable_scope(self.name, reuse=self.reuse):
            features = ops.feature_extraction(inputs,
                                              channels,
                                              scope='feature_extraction', is_training=self.is_training,
                                              bn_decay=None)

            H = ops.up_unit(features, self.up_ratio,
                            self.opts.upsampler,
                            scope="up_block",
                            is_training=self.is_training, bn_decay=None)

            coord = ops.conv2d(H, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])
            outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
                                  [inputs.shape[0], self.num_point * self.up_ratio, -1])  # B, N, 4, 3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs


class PUNET(object):
    """
    PU-Net:
    https://arxiv.org/abs/1801.06761
    """
    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs, bradius):
        num_point = inputs.get_shape()[1].value
        l0_xyz = inputs[:, :, 0:3]
        l0_normals = None  # do not use normals
        use_bn = False
        use_ibn = False
        bn_decay = None
        is_training = self.is_training

        with tf.variable_scope(self.name, reuse=self.reuse):
            # Layer 1
            l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_normals, npoint=num_point,
                                                               radius=bradius * 0.05,
                                                               bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer1')

            l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=num_point / 2,
                                                               radius=bradius * 0.1, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[64, 64, 128], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer2')

            l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=num_point / 4,
                                                               radius=bradius * 0.2, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[128, 128, 256], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer3')

            l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=num_point / 8,
                                                               radius=bradius * 0.3, bn=use_bn, ibn=use_ibn,
                                                               nsample=32, mlp=[256, 256, 512], mlp2=None,
                                                               group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay,
                                                               scope='layer4')

            # Feature Propagation layers
            up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay,
                                              scope='fa_layer1', bn=use_bn, ibn=use_ibn)

            up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay,
                                              scope='fa_layer2', bn=use_bn, ibn=use_ibn)

            up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay,
                                              scope='fa_layer3', bn=use_bn, ibn=use_ibn)

            concat_feat = tf.concat([up_l4_points, up_l3_points, up_l2_points, l1_points, l0_xyz], axis=-1)
            concat_feat = tf.expand_dims(concat_feat, axis=2)

            # concat feature
            if self.opts.upsampler == 'original':
                with tf.variable_scope('up_layer', reuse=tf.AUTO_REUSE):
                    new_points_list = []
                    for i in range(self.up_ratio):
                        concat_feat = ops.conv2d(concat_feat, 256, [1, 1],
                                                 padding='VALID', stride=[1, 1],
                                                 bn=False, is_training=is_training,
                                                 scope='fc_layer0_%d' % (i), bn_decay=bn_decay)

                        new_points = ops.conv2d(concat_feat, 128, [1, 1],
                                                padding='VALID', stride=[1, 1],
                                                bn=use_bn, is_training=is_training,
                                                scope='conv_%d' % (i),
                                                bn_decay=bn_decay)
                        new_points_list.append(new_points)
                    net = tf.concat(new_points_list, axis=1)
            else:
                net = ops.up_unit(concat_feat, self.up_ratio,
                                  self.opts.upsampler,
                                  scope="up_block",
                                  use_att=self.opts.use_att,
                                  is_training=self.is_training, bn_decay=None)

            # get the xyz
            coord = ops.conv2d(net, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            coord = tf.squeeze(coord, [2])  # B*(2N)*3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return coord


class PUGCN(object):
    """
    PU-GCN: Point Cloud Upsampling using Graph Convolutional Networks
    https://arxiv.org/abs/1912.03264.pdf
    """

    def __init__(self, opts, is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + self.opts.more_up
        self.out_num_point = int(self.num_point * self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            features, idx = ops.feature_extractor(inputs,
                                                  self.opts.block, self.opts.n_blocks,
                                                  self.opts.channels, self.opts.k, self.opts.d,
                                                  use_global_pooling=self.opts.use_global_pooling,
                                                  scope='feature_extraction', is_training=self.is_training,
                                                  bn_decay=None)

            H = ops.up_unit(features, self.up_ratio_real,
                            self.opts.upsampler,
                            k=self.opts.k,
                            idx=idx,
                            scope="up_block",
                            use_att=self.opts.use_att,
                            is_training=self.is_training, bn_decay=None)

            coord = ops.conv2d(H, 32, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None,
                               activation_fn=tf.nn.leaky_relu
                               )

            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            outputs = tf.squeeze(coord, [2])

            if self.up_ratio_real > self.up_ratio:
                outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))
            outputs += tf.reshape(tf.tile(tf.expand_dims(inputs, 2), [1, 1, self.up_ratio, 1]),
                                  [inputs.shape[0], self.num_point * self.up_ratio, -1])  # B, N, 4, 3

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs

