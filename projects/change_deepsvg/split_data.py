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

black_box = torch.tensor([
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  96.],
    [  1.,   0.,  96.,   0.,   0.,   0.,   0.,   0., 191.],
    [  1.,   0., 191.,   0.,   0.,   0.,   0.,  96., 191.],
    [  1.,  96., 191.,   0.,   0.,   0.,   0., 191., 191.],
    [  1., 191., 191.,   0.,   0.,   0.,   0., 191.,  96.],
    [  1., 191.,  96.,   0.,   0.,   0.,   0., 191.,   0.],
])

def process_file():
    svgo_folder = "/zecheng/svg/icon-shop/convert-p"
    new_meta = []
    for line in content:
        file_name = os.path.basename(line["new_path"])
        new_line = line.copy()
        new_line["convert_path"] = os.path.join(svgo_folder, file_name)
        new_meta.append(new_line)
    auto_save_data(new_meta, "/zecheng/svg/icon-shop/svgo_meta.jsonl")


def split_path(svg_file):
    svg = SVG.load_svg(svg_file)
    svg_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
    svg_tensors = torch.cat(svg_tensors)
    if svg_tensors[:6].equal(black_box):
        svg_tensors = svg_tensors[6:]
    svg_tensors = [svg_tensors]
    svg = SVGTensor.from_data(torch.cat(svg_tensors))
    svg = SVG.from_tensor(svg.data, viewbox=Bbox(200))
    svg.translate(Point(4, 4))
    svg.fill_(True)

def main(content, max_filter):
    cnt = 0
    final_res = []
    for line in content:
        svg_path = line.get("convert_path")
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
    # process_file()
    # exit()
    max_filter = 10000
    main(content, max_filter)


