import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))

# if __name__ == '__main__':
#     with tf.Session() as sess:
#         g = tf.get_default_graph()
#         x = tf.constant([10., 2.])
#         with g.gradient_override_map({'Relu': 'GuidedRelu'}):
#             y = tf.nn.relu(x)
#             z = tf.reduce_sum(-y ** 2)
#         tf.global_variables_initializer().run()
#         print x.eval(), y.eval(), z.eval(), tf.gradients(z, x)[0].eval()

if __name__ == '__main__':
    with tf.Session() as sess:
        g = tf.get_default_graph()
        x = tf.constant([10., 2.])
        # with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        y = tf.nn.relu(x)
        z = tf.reduce_sum(-y ** 2)
        tf.initialize_all_variables().run()
        print x.eval(), y.eval(), z.eval(), tf.gradients(z, y)[0].eval() , tf.gradients(z, x)[0].eval()