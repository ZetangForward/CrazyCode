import os
import glob
import logging
import json
import sys
sys.path.append("/workspace/zecheng/modelzipper/projects")
from tqdm import tqdm
from concurrent import futures
import argparse
from change_deepsvg.svglib.svg import SVG
from change_deepsvg.svglib.geom import Bbox, Angle, Point
from change_deepsvg.difflib.tensor import SVGTensor
from queue import Queue 
import torch
from PIL import Image  
import IPython.display as ipd
from modelzipper.tutils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, default=None)
    parser.add_argument("--ncols", '-nc', type=int, default=6)
    parser.add_argument("--nrows", '-nr', type=int, default=6)
    parser.add_argument("--output_dir", "-od", type=str, default=None)
    parser.add_argument("--metadata_dir", "-md", type=str, default=None)
    parser.add_argument("--model_tag", "-mt", type=str, default="llama")
    parser.add_argument("--add_color", "-ac", action="store_true")
    parser.add_argument("--single_save", "-ss", action="store_true")
    
    args = parser.parse_args()
    
    generated_res = auto_read_data(args.file)   
    
    print(f"Total {len(generated_res)} samples")

    file_name = os.path.basename(args.file).split(".")[0]
    
    ## convert batch svgs
    all_svgs = []
    cnt = 0
    for line in generated_res:
        cnt += 1
        try:
            gen_svg = SVG.from_str(line["processed_res"][0])
            golden_svg = SVG.from_str(line["golden_svg"])
            kwords = line["prompt"].split(":")[1].split("#")[0].strip()
            # render 
            center = Point(0, 0)  # Point(-4, -4)
            gen_img = gen_svg.draw(do_display=False, return_png=True)
            if args.add_color:
                gen_img = gen_svg.translate(center).normalize().split_paths().set_color("random").draw(do_display=False, return_png=True)
            else:
                gen_img = gen_svg.translate(center).normalize().split_paths().draw(do_display=False, return_png=True)
            golden_img = golden_svg.draw(do_display=False, return_png=True)
            
            all_svgs.append([gen_img, golden_img, kwords])
        except Exception as e:
            print(f"Error in {cnt}-th sample, {e}")
            continue
    batch_size = args.ncols * args.nrows
    
    print_c("Start to save images")
    if not os.path.exists(args.output_dir):
        print_c("Output dir not exists, create one")
        os.makedirs(args.output_dir)
    
    if args.single_save:
        print_c("Single save mode, save to {}".format(args.output_dir))
        
        meta_data_file = os.path.join(args.metadata_dir, f"{args.model_tag}_{file_name}_meta.jsonl")
        if os.path.exists(meta_data_file):
            print_c(f"Meta data file {meta_data_file} already exists, skip")
            meta_data = auto_read_data(meta_data_file)
        else:
            meta_data = []
        for i, item in enumerate(all_svgs):
            gen_image, golden_image, caption  = item[0], item[1], item[2]

            tmp = "-".join([i.strip() for i in caption.split(",")][:5])
            saved_file_name = f"{i}_{tmp}.png"
            gen_save_path = os.path.join(args.output_dir, f"{i}_gen_{tmp}.png")
            golden_save_path = os.path.join(args.output_dir, f"{i}_golden_{tmp}.png")
            gen_image.save(gen_save_path)
            golden_image.save(golden_save_path)
            meta_data.append({
                "gen": gen_save_path,
                "golden": golden_save_path,
                "caption": caption,
            })
            
            if i % 50 == 0:
                print(f"Single Save mode | Saved {i} images to {args.output_dir}")
        auto_save_data(meta_data, meta_data_file)
        print(f"Done! length of meta data: {len(meta_data)}, saved to {meta_data_file} | number of images in dir: {count_png_files(args.output_dir)}")

    else:
        print_c("Batch save mode, save to {}".format(args.output_dir))
        for i in range(0, len(all_svgs), batch_size):
            if i + args.ncols * args.nrows > len(all_svgs):
                break

            batch_svgs = all_svgs[i:i+args.ncols*args.nrows]
            gen_imgs = dict([(i, [None, item[0]]) for i, item in enumerate(batch_svgs)])
            golden_imgs = dict([(i, [None, item[1]]) for i, item in enumerate(batch_svgs)])
            caption_imgs = [item[2] + "\n" for i, item in enumerate(batch_svgs)]
            
            print(f"Batch {i}-{i+args.ncols*args.nrows} generating images, save to {args.model_tag}_{file_name}_gen_{i}-{i+args.ncols*args.nrows}.png")
            
            visualize_batch_images(gen_imgs, ncols=args.ncols, nrows=args.nrows, subplot_size=2, 
                                output_file=os.path.join(args.output_dir, f"{args.model_tag}_{file_name}_gen_{i}-{i+args.ncols*args.nrows}.png"))
            visualize_batch_images(golden_imgs, ncols=args.ncols, nrows=args.nrows, subplot_size=2, 
                                output_file=os.path.join(args.output_dir, f"{args.model_tag}_{file_name}_golden_{i}-{i+args.ncols*args.nrows}.png"))
            with open(os.path.join(args.output_dir, f"{args.model_tag}_{file_name}_caption_{i}-{i+args.ncols*args.nrows}.txt"), "w") as f:
                f.writelines(caption_imgs)
        print("Done!")