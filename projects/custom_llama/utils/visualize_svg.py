import numpy as np
import os
import glob
import logging
import json
import sys
sys.path.append("/workspace/zecheng/modelzipper/projects")
from tqdm import tqdm
from concurrent import futures
from argparse import ArgumentParser
from change_deepsvg.svglib.svg import SVG
from change_deepsvg.svglib.geom import Bbox, Angle, Point
from change_deepsvg.difflib.tensor import SVGTensor
from modelzipper.tutils import *
import torch
from tqdm import trange
from PIL import Image


def convert_svg(t, colored=False):
    svg = SVGTensor.from_data(t)
    svg = SVG.from_tensor(svg.data, viewbox=Bbox(200))
    svg.numericalize(n=200)
    if colored:
        svg = svg.split_paths().set_color("random")
    str_svg = svg.to_str()
    return svg, str_svg


def merge_images(
        folder_path, image_suffix, num_images, raw_image_size_w=None,
        raw_image_size_h=None, image_row=10, image_col=10, save_dir=None,
    ):
    image_list = []
    for i in range(num_images):
        filename = f'{i}_{image_suffix}'
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image_list.append(image)

    sub_image_w = raw_image_size_w if raw_image_size_w is not None else image_list[0].size[0]
    sub_image_h = raw_image_size_h if raw_image_size_h is not None else image_list[0].size[1]

    big_image_size = (sub_image_w * 10, sub_image_h * 10)
    big_image = Image.new('RGB', big_image_size)
    big_images = []

    for i, image in enumerate(image_list):
        if i >= image_row * image_col:
            i -= image_row * image_col

        row = i // image_row
        col = i % image_col
        big_image.paste(image, (col * image.size[0], row * image.size[1]))
        
        if (i + 1) % (image_row * image_col) == 0:
            big_images.append(big_image)
            big_image = Image.new('RGB', big_image_size)

    if save_dir is not None:
        for i, big_image in enumerate(big_images):
            save_path = os.path.join(save_dir, f'big_map_{i}_{image_suffix}')
            big_image.save(save_path)
            print_c(f"save big map {i} to {save_path}")
    return big_images


def main(cl: int = 0):
    assert cl in [1, 2, 3], "compress level must be 1, 2, 3"
    print_c(f"visualize compress level: {cl}, begin!", "magenta")


    ROOT_DIR = "/zecheng2/vqllama/test_vqllama_quantizer/test_0"
    COMPRESS_LEVEL = cl
    FILE_PATH = os.path.join(ROOT_DIR, f"compress_level_{COMPRESS_LEVEL}_predictions.pkl")
    
    SAVE_DIR = auto_mkdir(os.path.join(ROOT_DIR, f"visualized_compress_level_{COMPRESS_LEVEL}"))
    BIG_MAP_SAVED_DIR = auto_mkdir(os.path.join(SAVE_DIR, "big_map")) # save big picture map
    SINGLE_IMAGE_SAVED_DIR = auto_mkdir(os.path.join(SAVE_DIR, "single_image")) # save single image    
    PATH_SAVED_PATH = os.path.join(SAVE_DIR, "svg_paths.jsonl") # save svg path

    DIRECT_GENERATE_BIG_MAP = True
    DIRECT_GENERATE_SINGLE_IMAGE = True


    if DIRECT_GENERATE_SINGLE_IMAGE:
        results = auto_read_data(FILE_PATH)
        keys = ['raw_predict', 'p_predict', 'golden', 'zs', 'xs_quantised']
        num_svgs = len(results[keys[0]])
        str_paths = []

        for i in trange(num_svgs):
            raw_predict = results['raw_predict'][i]
            p_predict = results['p_predict'][i]
            golden = results['golden'][i]

            p_svg, p_svg_str = convert_svg(p_predict, True)
            g_svg, g_svg_str = convert_svg(golden, True)

            str_paths.append({
                "p_svg_str": p_svg_str,
                "g_svg_str": g_svg_str,
            })
            
            p_svg.save_png(os.path.join(SINGLE_IMAGE_SAVED_DIR, f"{i}_p_svg.png"))
            g_svg.save_png(os.path.join(SINGLE_IMAGE_SAVED_DIR, f"{i}_g_svg.png"))
        
        auto_save_data(str_paths, PATH_SAVED_PATH)

    if DIRECT_GENERATE_BIG_MAP:
        p_svg_images = merge_images(
            folder_path=SINGLE_IMAGE_SAVED_DIR, 
            image_suffix='p_svg.png', 
            num_images=2000, 
            save_dir=BIG_MAP_SAVED_DIR
        )
        g_svg_images = merge_images(
            folder_path=SINGLE_IMAGE_SAVED_DIR, 
            image_suffix='g_svg.png', 
            num_images=2000, 
            save_dir=BIG_MAP_SAVED_DIR
        )

    
if __name__ == "__main__":
    fire.Fire(main)


