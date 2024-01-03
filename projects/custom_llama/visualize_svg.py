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


def convert_svg(t, colored=False):
    svg = SVGTensor.from_data(t)
    svg = SVG.from_tensor(svg.data, viewbox=Bbox(200))
    str_svg = svg.to_str()
    if colored:
        svg = svg.draw_colored()
    return svg, str_svg
  




def main():
    FILE = "/zecheng2/vqllama/test_vqllama_quantizer/test_0/predictions.pkl"
    results = auto_read_data(FILE)
    keys = ['raw_predict', 'p_predict', 'golden']
    num_svgs = len(results[keys[0]])

    for i in range(num_svgs):
        raw_predict = results['raw_predict'][i]
        p_predict = results['p_predict'][i]
        golden = results['golden'][i]
        
        p_svg, p_svg_str = convert_svg(p_predict, True)
        g_svg, g_svg_str = convert_svg(golden, True)
        
        import pdb; pdb.set_trace()
        p_svg.save_png(f"p_svg_{i}.png")
        g_svg.save_png(f"g_svg_{i}.png")
        


if __name__ == "__main__":
    main()


