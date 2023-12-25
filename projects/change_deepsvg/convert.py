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

black_box = torch.tensor([
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  96.],
    [  1.,   0.,  96.,   0.,   0.,   0.,   0.,   0., 191.],
    [  1.,   0., 191.,   0.,   0.,   0.,   0.,  96., 191.],
    [  1.,  96., 191.,   0.,   0.,   0.,   0., 191., 191.],
    [  1., 191., 191.,   0.,   0.,   0.,   0., 191.,  96.],
    [  1., 191.,  96.,   0.,   0.,   0.,   0., 191.,   0.],
])


def build_mesh_data(svg_file):
    try:

        svg = SVG.load_svg(svg_file)
        svg.numericalize(n=200)
        svg_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
        svg_tensors = torch.cat(svg_tensors)
        if svg_tensors[:6].equal(black_box):
            svg_tensors = svg_tensors[6:]
        return svg_tensors

    except:
        print_c("Error in build_mesh_data", 'red')
        return None

def convert_to_mesh(mesh_data, num_sub_path = 3):
    idx = 0
    pair_sub_paths = []
    new_mesh_data = mesh_data.copy()
    for i in range(0, len(mesh_data), num_sub_path):
        if i + num_sub_path <= len(mesh_data):
            pair_sub_paths.append([k for k in range(idx, idx + num_sub_path)])
            idx += num_sub_path
        elif i < len(mesh_data) and len(mesh_data) - i < num_sub_path:
            pair_sub_paths.append([k for k in range(idx, idx + num_sub_path)])
            new_mesh_data = torch.cat([new_mesh_data, torch.zeros(num_sub_path - (len(mesh_data) - i), 9)])
        else:
            break
    return new_mesh_data, pair_sub_paths


def convert_svg(svg_file):
    try:
        filename = os.path.splitext(os.path.basename(svg_file))[0]

        svg = SVG.load_svg(svg_file)
        # make sure the argument is integer
        svg.numericalize(n=200)
        svg_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)

        svg_tensors = torch.cat(svg_tensors)
        if svg_tensors[:6].equal(black_box):
            svg_tensors = svg_tensors[6:]

        # FIGR-8-SVG specific operation
        svg_tensors = torch.cat(svg_tensors)
        if svg_tensors[:6].equal(black_box):
            svg_tensors = svg_tensors[6:]
        svg_tensors = [svg_tensors]
        svg = SVGTensor.from_data(torch.cat(svg_tensors))
        svg = SVG.from_tensor(svg.data, viewbox=Bbox(200))
        svg.translate(Point(4, 4))
        svg.fill_(True)
        svg.save_svg('test/{}.svg'.format(filename))
    except:
        return


def main(args):

    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        svg_files = glob.glob(os.path.join(args.data_folder, "*.svg"))
        meta_data = []

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [executor.submit(convert_svg, svg_file, meta_data)
                                   for svg_file in svg_files]

            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)

    identifier = 'FIGR-SVG-svgo-meta'
    with open(f'{identifier}.json', 'w') as f:
        f.write(json.dumps(meta_data))

    logging.info("SVG Preprocessing complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    data_foler = '/zecheng/svg/icon-shop/meta_data_filter_750.jsonl'
    meta_data = auto_read_data(data_foler)

    saved_ = []
    for i in trange(len(meta_data)):
        sample = meta_data[i]
        svg_file, keywords = sample['file_path'], sample['keywords']
        mesh_data = build_mesh_data(svg_file)
        # import pdb; pdb.set_trace()
        saved_.append({'keywords': keywords, 'mesh_data': mesh_data})
    
    auto_save_data(saved_, '/zecheng/svg/icon-shop/mesh_data_svg_convert_p.pkl')


    # parser = ArgumentParser()
    # parser.add_argument("--data_folder", default=os.path.join("dataset", data_foler))
    # parser.add_argument("--workers", default=32, type=int)

    # args = parser.parse_args()

    # main(args)
