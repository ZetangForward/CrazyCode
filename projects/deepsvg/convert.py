import os
import glob
import logging
import json

from tqdm import tqdm
from concurrent import futures
from argparse import ArgumentParser
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.geom import Bbox, Angle, Point
from deepsvg.difflib.tensor import SVGTensor

import torch

black_box = torch.tensor([
    [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  96.],
    [  1.,   0.,  96.,   0.,   0.,   0.,   0.,   0., 191.],
    [  1.,   0., 191.,   0.,   0.,   0.,   0.,  96., 191.],
    [  1.,  96., 191.,   0.,   0.,   0.,   0., 191., 191.],
    [  1., 191., 191.,   0.,   0.,   0.,   0., 191.,  96.],
    [  1., 191.,  96.,   0.,   0.,   0.,   0., 191.,   0.],
])


def convert_svg(svg_file):
    try:
        filename = os.path.splitext(os.path.basename(svg_file))[0]

        svg = SVG.load_svg(svg_file)

        # make sure the argument is integer
        # svg.numericalize(n=200)

        svg_tensors = svg.to_tensor(concat_groups=False, PAD_VAL=0)
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

    data_foler = 'FIGR-SVG-svgo'
    
    parser = ArgumentParser()
    parser.add_argument("--data_folder", default=os.path.join("dataset", data_foler))
    parser.add_argument("--workers", default=32, type=int)

    args = parser.parse_args()

    main(args)
