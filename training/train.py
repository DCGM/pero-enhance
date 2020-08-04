# -*- coding: utf-8 -*-
import shutil
import time
import os
import argparse
import re
import pickle

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

import tf_nets
import enhancement_dataset

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hq-lines', required=True, help='Paths to high quality line images with their respective transcriptions.')
    parser.add_argument('--lq-lines', required=True, help='Paths to low quality line images with their respective transcriptions.')
    parser.add_argument('--tst-lines', required=True, help='Paths to test line images with their respective transcriptions.')
    parser.add_argument('-o', '--output-path', required=True, help='Output path')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint to restore')
    parser.add_argument('--degrade-trn', action='store_true', help='Adds further specified random synthetic degradation to training lq data.')
    parser.add_argument('--degrade-tst', action='store_true', help='Adds further specified random synthetic degradation to testing data.')
    parser.add_argument('--max-width', type=int, default=800, help='Clip image data to this length.')
    parser.add_argument('--line-height', type=int, default=32, help='Target line height.')
    parser.add_argument('--ocr-iterations', type=int, default=20000)
    parser.add_argument('--max-iterations', type=int, default=500001)
    parser.add_argument('--view-step', type=int, default=1000, help='Save some outputs after this number of training steps.')
    parser.add_argument('--num-test', type=int, default=100, help='How many testing images to save.')
    parser.add_argument('--ctc-lambda', type=float, default=1, help='CTC training loss weight.')
    parser.add_argument('--color-lambda', type=float, default=0, help='Color consistency training loss weight.')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--mask-prob', type=float, default=0, help='Probability of masking the image during synthetic degradation.')
    parser.add_argument('--bin-prob', type=float, default=0, help='Probability of binarization the image during synthetic degradation.')
    parser.add_argument('--blur-prob', type=float, default=0, help='Probability of blurring the image during synthetic degradation.')
    parser.add_argument('--noise-prob', type=float, default=0, help='Probability of noising the image during synthetic degradation.')
    parser.add_argument('--features', type=int, default=32, help='Number of features of the first convolutional layer.')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help="If not set setGPU is called. Set this to 0 on desktop. Leave it empty on SGE.")
    return parser.parse_args()

def print_and_show_transcriptions(output, chars, gt=None, batch_size=16):
    result_strings = []
    for i in range(batch_size):
        s = []
        if output is not None:
            pos, = np.nonzero(output[0].indices[:, 0] == i)
            if pos.size:
                for val in output[0].values[pos]:
                    s.append(chars[val])
        result_strings.append(''.join(s))

    gt_strings = []
    for i in range(batch_size):
        s = []
        if gt is not None:
            pos, cc, _ = gt
            for p, c in zip(pos, cc):
                if p[0] == i:
                    s.append(chars[c])
        gt_strings.append(''.join(s))

    for g, o in zip(gt_strings, result_strings):
        print('{:>60} --- {}'.format(g, o))

def main():

    args = parseargs()

    train_dataset_hq = enhancement_dataset.EnhancementDataset(
        lines_path = args.hq_lines,
        batch_size = args.batch_size,
        max_width = args.max_width,
        line_height = args.line_height
    )
    train_dataset_lq = enhancement_dataset.EnhancementDataset(
        lines_path = args.lq_lines,
        batch_size = args.batch_size,
        mask_prob = args.mask_prob,
        bin_prob = args.bin_prob,
        blur_prob = args.blur_prob,
        noise_prob = args.noise_prob,
        max_width = args.max_width,
        line_height = args.line_height
    )
    test_dataset = enhancement_dataset.EnhancementDataset(
        lines_path = args.tst_lines,
        batch_size = 1,
        mask_prob = args.mask_prob,
        bin_prob = args.bin_prob,
        blur_prob = args.blur_prob,
        noise_prob = args.noise_prob,
        max_width = args.max_width,
        line_height = args.line_height
    )

    ugan = tf_nets.transformer_gan(
            init_features=args.features,
            data_shape=(args.line_height, args.max_width),
            batch_size=args.batch_size,
            max_labels=100,
            ctc_lambda=args.ctc_lambda,
            color_lambda=args.color_lambda,
            num_chars=len(train_dataset_hq.chars))

    print('got data and net')

    if not os.path.exists(os.path.join(args.output_path, 'model')):
        os.makedirs(os.path.join(args.output_path, 'model'))
    if not os.path.exists(os.path.join(args.output_path, 'images')):
        os.makedirs(os.path.join(args.output_path, 'images'))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if args.gpu_id is None:
        tf_nets.setGPU()
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    config = tf.ConfigProto()

    start_iteration = 0
    with tf.Session(graph=ugan.graph, config=config) as session:
        ugan.build_optimizers(session)
        if args.checkpoint:
            ugan.saver.restore(session, args.checkpoint)
            print('\n\n\nRestored model {}.\n'.format(args.checkpoint))
        else:
            print('\n\n\nNo model restored, training from scratch.\n')

        # OCR pretraining
        for iteration in range(args.ocr_iterations):
            if np.random.rand()>0.5:
                images, _, ctc_targets, seq_lengths = train_dataset_hq.get_batch(degrade=False)
            else:
                images, _, ctc_targets, seq_lengths = train_dataset_lq.get_batch(degrade=args.degrade_trn)
            feed_dict = {ugan.input_images: images,
                         ugan.ctc_targets: ctc_targets,
                         ugan.ctc_seq_len: seq_lengths}
            if iteration % 100 == 0:
                ctc_loss, ctc_cer, ctc_decoded, ctc_targets = session.run(
                        [ugan.ctc_loss, ugan.ctc_cer, ugan.ctc_decoded, ugan.ctc_targets],
                        feed_dict=feed_dict)
                print_and_show_transcriptions(ctc_decoded, train_dataset_hq.chars,
                                              ctc_targets, batch_size=train_dataset_hq.batch_size)
            else:
                _, ctc_loss, ctc_cer = session.run([ugan.ocr_optimizer, ugan.ctc_loss, ugan.ctc_cer],
                                                   feed_dict=feed_dict)
            print('step', iteration, ', CTC: {:2.2f}, CER: {:2.2f}'.format(ctc_loss, ctc_cer))
        ugan.saver.save(session, os.path.join(args.output_path, 'model', 'ugan_checkpoint_ocr_pretrain'))

        # Restoration training
        for iteration in range(args.max_iterations):
            input_images, transcriptions, ctc_targets, seq_lengths = train_dataset_lq.get_batch(degrade=args.degrade_trn)
            reference_images, reference_transcriptions, _, _ = train_dataset_hq.get_batch(degrade=False)
            feed_dict = {ugan.input_images: input_images,
                         ugan.transcriptions: transcriptions,
                         ugan.ctc_targets: ctc_targets,
                         ugan.ctc_seq_len: seq_lengths,
                         ugan.reference_images: reference_images,
                         ugan.reference_transcriptions: reference_transcriptions
                         }
            _, _, l_dis, l_gen, ctc_gen_loss, ctc_gen_decoded, ctc_gen_cer = session.run(
                [ugan.dis_optimizer, ugan.gen_optimizer, ugan.d_loss, ugan.g_loss,
                 ugan.ctc_gen_loss, ugan.ctc_gen_decoded, ugan.ctc_gen_cer],
                feed_dict=feed_dict)
            print('step', iteration, ', gen loss: {:2.2f}, disc loss: {:2.2f}, CTC: {:2.2f}, CER: {:2.2f}'.format(
                l_gen, l_dis, ctc_gen_loss, ctc_gen_cer))

            if iteration % args.view_step == 0 and iteration > 0:
                print('saving test outputs')
                ugan.saver.save(session, os.path.join(args.output_path, 'model', 'ugan_checkpoint_{:06d}'.format(iteration)))
                image_output_path = os.path.join(args.output_path, 'images', 'step_{:06d}'.format(iteration))

                if not os.path.exists(image_output_path):
                    os.makedirs(image_output_path)

                for ind in range(args.num_test):
                    input_images, transcriptions, _, _ = test_dataset.get_batch(degrade=args.degrade_tst)
                    feed_dict = {ugan.inference_content: input_images,
                                 ugan.inference_transcriptions: transcriptions}
                    output = session.run(ugan.inference_op, feed_dict=feed_dict)
                    output = np.clip(output[0,:,:,:], 0, 1)
                    cv2.imwrite(os.path.join(image_output_path, 'img_{:04d}_out.jpg'.format(ind)),
                                    (255*output).astype(np.uint8))
                    cv2.imwrite(os.path.join(image_output_path, 'img_{:04d}_in.jpg'.format(ind)),
                                    (255*input_images[0,:,:,:]).astype(np.uint8))


if __name__=='__main__':
    main()
