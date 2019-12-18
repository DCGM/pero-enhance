# -*- coding: utf-8 -*-
import shutil
import time
import os
import argparse
import re
import configparser

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import matplotlib.pyplot as plt
import cv2
import shapely.geometry
import tkinter as tk

from pero.document_ocr import layout, page_parser
import repair_engine

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-images', required=True, help='Input image folder to enhance')
    parser.add_argument('-o', '--output-path', required=True, help='output image folder to save enhanced images')
    parser.add_argument('-r', '--repair-json', help='Path to repair engine json', default='./model/enhance_LN_2019-12-18/repair_engine.json')
    parser.add_argument('-x', '--input-page', help='Input page xml folder (if empty or None, automatic line detection and OCR is run', default=None)
    parser.add_argument('-p', '--parse-config', help='Path to page parser config file', default=None)
    return parser.parse_args()

def main():

    args = parseargs()
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    print('Loading engines...')
    enhancer = repair_engine.EngineRepairCNN(args.repair_json)

    if args.parse_config is not None:
        config = configparser.ConfigParser()
        config.read(args.parse_config)
        # convert relative paths to absolute
        for section, key in [['LINE_PARSER', 'MODEL_PATH'], ['OCR', 'OCR_JSON']]:
            if not os.path.isabs(config[section][key]):
                config[section][key] = os.path.realpath(os.path.join(os.path.dirname(args.parse_config), config[section][key]))
        parser = page_parser.PageParser(config)
    else:
        parser = None

    for filename in os.listdir(args.input_images):
        page_img = cv2.imread(os.path.join(args.input_images, filename))
        page_id, _ = os.path.splitext(filename)
        page_xml_file = os.path.join(args.input_page, page_id+'.xml')

        if os.path.exists(page_xml_file):
            page_layout = layout.PageLayout(file=page_xml_file)
        elif not os.path.exists(page_xml_file) and parser is not None:
            print('Page xml file for page {} not found, running automatic parser...'.format(page_id))
            page_layout = layout.PageLayout(id=page_id, page_size=(page_img.shape[0], page_img.shape[1]))
            page_layout = parser.process_page(page_img, page_layout)
        else:
            raise Exception('Page xml file for page {} not found and automatic page parser config not specified.'.format(page_id))

        page_img = enhancer.enhance_page(page_img, page_layout)

        cv2.imwrite(os.path.join(args.output_path, '{}_enhanced.jpg'.format(page_id)), page_img)

if __name__=='__main__':
    main()
