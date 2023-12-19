import os
import glob
import logging
import json
import sys
sys.path.append("/workspace/zecheng/modelzipper/projects")
from modelzipper.tutils import *
from tqdm import tqdm
from concurrent import futures
from argparse import ArgumentParser
from change_deepsvg.svglib.svg import SVG
from change_deepsvg.svglib.geom import Bbox, Angle, Point
from change_deepsvg.difflib.tensor import SVGTensor

import torch

def process_file():
    svgo_folder = "/zecheng/svg/icon-shop/svgo-p"
    new_meta = []
    for line in content:
        file_name = os.path.basename(line["ori_path"])
        new_line = line.copy()
        new_line["svgo_path"] = os.path.join(svgo_folder, file_name)
        new_meta.append(new_line)
    auto_save_data(new_meta, "/zecheng/svg/icon-shop/svgo_meta.jsonl")


def split_path(svg_file):
    filename = os.path.splitext(os.path.basename(svg_file))[0]
    svg = SVG.load_svg(svg_file)
    svg_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
    import pdb; pdb.set_trace()

def main(content, max_filter):
    cnt = 0
    final_res = []
    for line in content:
        svg_path = line.get("svg_path")
        print(svg_path)
        tmp = split_path(svg_path)
        final_res.extend(tmp)
        cnt += 1
        if cnt >= max_filter:
            break
    
    ### save file




if __name__ == "__main__":
    meta_file = "/zecheng/svg/icon-shop/svgo_meta.jsonl"
    content = auto_read_data(meta_file)
    max_filter = 10000
    main(content, max_filter)


