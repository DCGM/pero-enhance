import numpy as np
import tensorflow as tf
import subprocess
import os
import sys
import transformer.model.transformer as tft
import transformer.model.embedding_layer as tf_embedding
import transformer.model.model_params as tf_params


def setGPU():
    freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    freeGpu = freeGpu.decode().strip()
    if len(freeGpu) == 0:
        print('no free GPU!')
        sys.exit(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu
    print('got GPU ' + (freeGpu))

    return(int(freeGpu.strip()))


class transformer_gan(object):

    def __init__(self, init_features=16, data_shape=(32,800), gp_scale=10,
                 batch_size=16, max_labels=100, num_chars=276, color_lambda=0,
                 ctc_lambda=1):

        self.num_chars = num_chars
        self.max_labels = max_labels
        self.batch_size = batch_size
        self.init_features = init_features
        self.scale = gp_scale
        self.ctc_lambda = ctc_lambda
        self.color_lambda = color_lambda
        self.final_stage = np.ceil((np.log2(data_shape[0]) - 2)).astype(np.uint8) # number of maxpooling steps to achieve tensor of size 4 or less
        self.params = tf_params.MEDIUM_PARAMS

        self.graph = tf.Graph()
        tf.reset_default_graph()
        self.initialized_vars = []
        with self.graph.as_default():
            with tf.variable_scope("DIS"):
                self.dis_transformer = tft.Transformer(self.params, train=True)
            with tf.variable_scope("GEN"):
                self.gen_transformer = tft.Transformer(self.params, train=True)

            # set i/o
            self.input_images = tf.placeholder(tf.float32, shape=(batch_size, data_shape[0], data_shape[1], 3), name='input_content')
            self.reference_images = tf.placeholder(tf.float32, shape=(batch_size, data_shape[0], data_shape[1], 3))
            self.transcriptions = tf.placeholder(tf.int32, shape=(batch_size, max_labels), name='input_transcriptions')
            self.reference_transcriptions = tf.placeholder(tf.int32, shape=(batch_size, max_labels))

            self.ctc_targets = tf.sparse_placeholder(tf.int32, name='ctc_targets')
            self.ctc_seq_len = tf.placeholder(tf.int32, [None], name='ctc_seq_len')

            self.inference_content = tf.placeholder(tf.float32, shape=(1, data_shape[0], data_shape[1], 3), name='inference_content')
            self.inference_transcriptions = tf.placeholder(tf.int32, shape=(1, max_labels), name='inference_transcriptions')
            self.inference_op = self.generator(self.inference_content, self.inference_transcriptions, train=False)
            self.inference_op = tf.identity(self.inference_op, name='inference_op')

    def build_optimizers(self, session):
        with self.graph.as_default():
            self.d_loss, self.g_loss = self.compute_losses()

            # initialize all vars and add them to optimizers
            self.dis_variables = [var for var in tf.trainable_variables() if var.name.startswith('DIS')]
            self.gen_variables = [var for var in tf.trainable_variables() if var.name.startswith('GEN')]
            self.ocr_variables = [var for var in tf.trainable_variables() if var.name.startswith('OCR')]

            self.dis_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.0, beta2=0.99, name='dis_Adam').minimize(self.d_loss, var_list=self.dis_variables)
            self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.0, beta2=0.99, name='gen_Adam').minimize(self.g_loss, var_list=self.gen_variables)
            self.ocr_optimizer = tf.train.AdamOptimizer(learning_rate=0.0003, name='ocr_Adam').minimize(self.ctc_loss, var_list=self.ocr_variables)

            session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

    def get_embeddings(self, tensor):
        embedding_softmax_layer = tf_embedding.EmbeddingSharedWeights(self.num_chars, self.params["hidden_size"], method="gather")
        return embedding_softmax_layer(tensor)

    def instance_norm(self, tensor):
        num_features = tensor.shape[-1]

        mean = tf.reduce_mean(tensor, axis=[1,2], keepdims=True)
        variance = tf.reduce_mean(tf.square(tensor - mean), axis=[1,2], keepdims=True)
        tensor_norm = (tensor - mean) * tf.rsqrt(variance + 1e-6)

        return tensor_norm

    def discriminator(self, tensor, transcriptions):
        with tf.variable_scope("DIS"):
            for stage in range(self.final_stage):
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None, name='first_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None, name='second_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensor = tf.layers.max_pooling2d(tensor, 2, 2)
            # final scoring layers
            height = int(tensor.shape[1])
            data_embeddings = tf.layers.conv2d(tensor, self.params['hidden_size'], (height, 1), padding='valid', activation=None, name='final', reuse=tf.AUTO_REUSE)
            data_embeddings = data_embeddings[:,0,:,:]

            transcription_embeddings = self.get_embeddings(transcriptions)
            tensor = self.dis_transformer(transcription_embeddings, data_embeddings, train=True)
            tensor = tf.nn.leaky_relu(tensor)
            score = tf.layers.dense(tensor, 1, name='score', reuse=tf.AUTO_REUSE)
            score = tf.reduce_mean(score, axis=1)
        return score

    def ocr(self, tensor):
        with tf.variable_scope("OCR"):
            for stage in range(2):
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None,
                                          name='first_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None,
                                          name='second_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensor = tf.layers.max_pooling2d(tensor, 2, 2)
            # final scoring layers
            height = int(tensor.shape[1])
            net = tf.layers.conv2d(tensor, self.params['hidden_size'], (height, 1), padding='valid',
                                       activation=None, name='final', reuse=tf.AUTO_REUSE)
            net = tf.nn.leaky_relu(net)
            net = net[:,0,:,:]
            for i in range(2):
                net = tf.layers.conv1d(
                    net, filters=self.params['hidden_size'], kernel_size=3, padding='same',
                    name='out_{:d}'.format(i), reuse=tf.AUTO_REUSE)
                net = tf.nn.leaky_relu(net)
            logits = tf.layers.conv1d(net, self.num_chars + 1, 3,
                                      padding='same', name='ocr_logits', reuse=tf.AUTO_REUSE)
            print('OCR output', logits.shape)
            return logits

    def generator(self, tensor, transcriptions, train=True):
        tensors_to_append = list()
        with tf.variable_scope("GEN"):
            # content encoder
            for stage in range(self.final_stage):
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None, name='first_down_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None, name='second_down_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensors_to_append.append(tensor)
                tensor = tf.layers.max_pooling2d(tensor, 2, 2)
            height = int(tensor.shape[1])
            width = int(tensor.shape[2])
            data_embeddings = tf.layers.conv2d(tensor, self.params['hidden_size'], (height, 1), padding='valid', activation=None, name='bottleneck', reuse=tf.AUTO_REUSE)
            data_embeddings = data_embeddings[:,0,:,:]

            # text encoder
            transcription_embeddings = self.get_embeddings(transcriptions)
            tensor = self.gen_transformer(transcription_embeddings, data_embeddings, train=train)
            tensor = tf.reshape(tensor, (-1, width, height, tensor.shape[2] // height))
            tensor = tf.transpose(tensor, (0, 2, 1, 3))

            # output decoder
            for stage in range(self.final_stage-1, -1, -1):
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None, name='first_up_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensor = tf.layers.conv2d(tensor, self.init_features * (2 ** stage), 3, padding='same', activation=None, name='second_up_stage_' + str(stage), reuse=tf.AUTO_REUSE)
                tensor = tf.nn.leaky_relu(tensor)
                tensor = self.instance_norm(tensor)
                tensor = tf.image.resize_images(tensor, (tensors_to_append[stage].shape[1:3]))
                tensor = tf.concat((tensors_to_append[stage], tensor), axis=-1)

            # produce output
            tensor = tf.layers.conv2d(tensor, self.init_features, 3, padding='same', activation=None,
                                      name='prefinal', reuse=tf.AUTO_REUSE)
            tensor = tf.nn.leaky_relu(tensor)
            output = tf.layers.conv2d(tensor, 3, 3, padding='same', activation=None, name='GEN_final', reuse=tf.AUTO_REUSE)
        return output

    def color_consistency_loss(self, input, output, tiles=8):
        height = int(input.shape[1])
        width = int(input.shape[2]) // tiles

        input_avgs = tf.nn.avg_pool(input, (1, height, width, 1), (1, height, width, 1), padding='SAME')
        output_avgs = tf.nn.avg_pool(output, (1, height, width, 1), (1, height, width, 1), padding='SAME')

        return tf.losses.mean_squared_error(input_avgs, output_avgs)


    def compute_losses(self):
        # restoration model output
        gen_out = self.generator(self.input_images, self.transcriptions)

        # discriminator scores assigned to restored output and reference image
        self.dis_gen_score  = self.discriminator(gen_out, self.transcriptions)
        self.dis_real_score = self.discriminator(self.reference_images, self.reference_transcriptions)

        # text recognition model logits read from restored output and reference image
        self.ctc_gen_logits = self.ocr(gen_out)
        self.ctc_gen_loss = tf.reduce_mean(tf.nn.ctc_loss(self.ctc_targets, self.ctc_gen_logits, self.ctc_seq_len,
        ctc_merge_repeated=True, time_major=False))

        # text recognition model training with CTC loss
        self.ctc_logits = self.ocr(self.input_images)
        self.ctc_gen_decoded, self.ctc_gen_log_prob = tf.nn.ctc_greedy_decoder(
            tf.transpose(self.ctc_gen_logits, (1, 0, 2)), self.ctc_seq_len, merge_repeated=True)
        self.ctc_gen_cer = tf.reduce_mean(tf.edit_distance(
            tf.cast(self.ctc_gen_decoded[0], tf.int32), self.ctc_targets))

        self.ctc_decoded, self.ctc_log_prob = tf.nn.ctc_greedy_decoder(tf.transpose(self.ctc_logits, (1, 0, 2)),
                                                                       self.ctc_seq_len, merge_repeated=True)
        self.ctc_cer = tf.reduce_mean(tf.edit_distance(tf.cast(self.ctc_decoded[0], tf.int32), self.ctc_targets))
        self.ctc_loss = tf.reduce_mean(tf.nn.ctc_loss(self.ctc_targets, self.ctc_logits, self.ctc_seq_len,
                                                      ctc_merge_repeated=True, time_major=False))

        # color consistency loss
        self.color_loss = self.color_consistency_loss(self.input_images, gen_out)

        # discriminator and restoraiton model losses for training
        d_loss = - tf.reduce_mean(self.dis_gen_score) + tf.reduce_mean(self.dis_real_score)
        g_loss = tf.reduce_mean(self.dis_gen_score) + self.ctc_lambda * self.ctc_gen_loss + self.color_lambda * self.color_loss

        # add gradient penalty to discriminator loss
        epsilon = tf.random_uniform([], 0.0, 1.0)
        mixed_input = epsilon * self.reference_images + (1 - epsilon) * gen_out
        # transcriptions serve as int32 indices for gather op so lets just mix them by sampling from input and reference transcriptions with epsilon and 1-epsilon probability, respectively
        mixed_weights = tf.random_uniform(self.transcriptions.shape, 0.0, 1.0)
        mixed_transcriptions = tf.zeros(self.transcriptions.shape)
        mixed_transcriptions = self.reference_transcriptions * tf.cast(mixed_weights<=epsilon, tf.int32)
        mixed_transcriptions += self.transcriptions * tf.cast(mixed_weights>epsilon, tf.int32)
        mixed_score = self.discriminator(mixed_input, mixed_transcriptions)
        gp = tf.gradients(mixed_score, mixed_input)[0]
        gp = tf.sqrt(tf.reduce_sum(tf.square(gp), axis=[1,2,3]))
        gp = tf.reduce_mean(tf.square(gp - 1.0) * self.scale)

        return d_loss + gp, g_loss
