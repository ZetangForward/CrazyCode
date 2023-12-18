from change_deepsvg.config import _Config
from change_deepsvg import utils
from change_deepsvg.utils import Stats, TrainVars, Timer, Ema
import os
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
from datetime import datetime
from torch.utils.data import DataLoader
import argparse
import importlib
import wandb
import tqdm

from change_deepsvg.difflib.tensor import SVGTensor
from change_deepsvg.svglib.svg import SVG
from change_deepsvg.svglib.geom import Bbox

def train(cfg: _Config, desc="", log_dir="./logs", debug=False, resume=False, predict=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() and not debug else "cpu")

    print("Parameters")
    cfg.print_params()

    print("Loading dataset")
    dataset_load_function = importlib.import_module(cfg.dataloader_module).load_dataset
    train_dataset, valid_dataset = dataset_load_function(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
    # batch size should be no less than the total number of data
    valid_dataloader = DataLoader(valid_dataset, batch_size=min(cfg.batch_size, len(valid_dataset)), shuffle=True, drop_last=True,
                            num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
    model = cfg.make_model().to(device)

    if cfg.pretrained_path is not None:
        print(f"Loading pretrained model {cfg.pretrained_path}")
        utils.load_model(cfg.pretrained_path, model)
        # utils.load_ckpt(cfg.pretrained_path, model, cfg)

    train_cfg_dict = dict(cfg.values())
    model_cfg_dict = dict(cfg.model_cfg.values())
    model_cfg_dict.update(train_cfg_dict)

    if not debug and not predict:
        project_name = 'deepsvg_quantize_path'
        wandb.init(project = project_name, config = model_cfg_dict)

    stats = Stats(num_steps=cfg.num_steps, num_epochs=cfg.num_epochs, steps_per_epoch=len(train_dataloader),
                  stats_to_print=cfg.stats_to_print)
    train_vars = TrainVars()
    valid_vars = TrainVars()
    timer = Timer()

    stats.num_parameters = utils.count_parameters(model)
    print(f"#Parameters: {stats.num_parameters:,}")

    # Summary Writer
    current_time = datetime.now().strftime("%b%d_%H:%M")
    experiment_identifier = f"{desc}-{current_time}"

    # summary_writer = SummaryWriter(os.path.join(log_dir, "tensorboard", "debug" if debug else "full", experiment_identifier))
    checkpoint_dir = os.path.join(log_dir, "models", experiment_identifier)
    visualization_dir = os.path.join(log_dir, "visualization", experiment_identifier)

    cfg.set_train_vars(train_vars, train_dataloader)
    cfg.set_train_vars(valid_vars, valid_dataloader)

    # Optimizer, lr & warmup schedulers
    optimizers = cfg.make_optimizers(model)
    scheduler_lrs = cfg.make_schedulers(optimizers, epoch_size=len(train_dataloader))
    scheduler_warmups = cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    # EMA
    # ema = Ema(model, 0.999)
    # ema.register()

    loss_fns = [l.to(device) for l in cfg.make_losses()]

    if resume:
        ckpt_exists = utils.load_ckpt_list(checkpoint_dir, model, None, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)

    if resume and ckpt_exists:
        print(f"Resuming model at epoch {stats.epoch+1}")
        stats.num_steps = cfg.num_epochs * len(train_dataloader)
    else:
        # Run a single forward pass on the single-device model for initialization of some modules
        single_foward_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size // cfg.num_gpus, shuffle=True, drop_last=True,
                                      num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
        data = next(iter(single_foward_dataloader))
        model_args, params_dict = [data[arg].to(device) for arg in cfg.model_args], cfg.get_params(0, 0)
        model(*model_args, params=params_dict)

    if not debug:
        model = nn.DataParallel(model)

    if predict:
        log = evaluate(cfg, model, device, loss_fns, train_vars, valid_dataloader, "valid", 0, 0, visualization_dir, True)
        return

    epoch_range = utils.infinite_range(stats.epoch) if cfg.num_epochs is None else range(stats.epoch, cfg.num_epochs)
    for epoch in epoch_range:
        print(f"Epoch {epoch+1}")

        # data = {commands=xxx, args=xxx}
        for n_iter, data in enumerate(train_dataloader):
            step = n_iter + epoch * len(train_dataloader)
            
            if cfg.num_steps is not None and step > cfg.num_steps:
                return

            model.train()
            model_args = [data[arg].to(device) for arg in cfg.model_args]
            labels = data["label"].to(device) if "label" in data else None
            params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

            for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, cfg.optimizer_starts), 1):
                optimizer.zero_grad()

                # 验证 tokenization
                # torch.set_printoptions(profile='full')
                # indices = model.get_codebook_indices(*model_args)
                # print(indices[0].reshape(2, 62))
                # # print(indices[1].reshape(2, 62))
                # exit()

                # commands_y, args_y = model.decode(indices[0].unsqueeze(0))

                # quantize_visualization_dir = './test'
                # if not os.path.exists(quantize_visualization_dir):
                #     os.makedirs(quantize_visualization_dir)
                # file_path = os.path.join(quantize_visualization_dir, f"{step}.svg")

                # tensor_pred = SVGTensor.from_cmd_args(commands_y[0].cpu(), args_y[0].cpu())
                # svg_pred = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(24), allow_empty=True).normalize()
                # svg_pred.fill_(True)
                # svg_pred.draw(do_display=False, return_png=True, with_points=False, file_path=file_path)

                # # exit()
                # continue

                output = model(*model_args, params=params_dict)
                loss_dict = loss_fn(output, labels, weights=weights_dict)

                if step >= optimizer_start:
                    loss_dict["loss"].backward()
                    if cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                    optimizer.step()
                    # ema.update()
                    # if scheduler_lr is not None:
                    #     scheduler_lr.step(loss_dict["loss"])
                    if scheduler_warmup is not None:
                        scheduler_warmup.step(metrics=loss_dict["loss"])
                    # if step >= cfg.warmup_steps:
                    #     optimizer.param_groups[0]['lr'] = max(optimizer.param_groups[0]['lr'], 1e-6 * cfg.num_gpus)

                stats.update_stats_to_print("train", loss_dict.keys())
                stats.update("train", step, epoch, {
                    ("lr" if i == 1 else f"lr_{i}"): optimizer.param_groups[0]['lr'],
                    **loss_dict
                })

            stats.update("train", step, epoch, {
                **weights_dict,
                "time": timer.get_elapsed_time()
            })

            log = {}  # for wandb

            if step % cfg.log_every == 0 and step > 0:
                print(stats.get_summary("train"))
                # stats.write_tensorboard(summary_writer, "train")
                # summary_writer.flush()

                # wandb
                log = {
                    **log,
                    'epoch': epoch,
                    'iter': n_iter,
                    'lr': optimizer.param_groups[0]['lr'],
                    **loss_dict,
                    'loss_cmd_weight': weights_dict['loss_cmd_weight'],
                    'loss_args_weight': weights_dict['loss_args_weight'],
                    'loss_visibility_weight': weights_dict['loss_visibility_weight'],
                }

            if step % cfg.val_every == 0 and step > 0:
                model.eval()

                with torch.no_grad():
                    # Visualization
                    imgs = cfg.visualize(model, train_vars, step, visualization_dir, loss_fns)
                    # ema.apply_shadow()
                    valid_log = evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid", step, epoch, visualization_dir)
                    # ema.restore()

                log = {
                    **log,
                    **valid_log,
                    'train_image': [wandb.Image(img) for img in imgs]
                }
                timer.reset()

            if not debug:
                wandb.log(log)

            if not debug and step % cfg.ckpt_every == 0 and step > 0:
                utils.save_ckpt_list(checkpoint_dir, model, cfg, optimizers, scheduler_lrs, scheduler_warmups, stats, train_vars)


def evaluate(cfg, model, device, loss_fns, vars, dataloader, split, step, epoch, visualization_dir, predict=False):
    print(f"Evaluate on: {split}")
    
    if len(dataloader) == 0:
        print("evaluation len(dataloader) is 0")
        return

    model.eval()
    with torch.no_grad():

        # visualization
        imgs = cfg.visualize(model, vars, step, visualization_dir, loss_fns, split=split)
        log = {
            'valid_image': [wandb.Image(img) for img in imgs]
        }
        
        # reconstruction error
        if not predict:
            valid_loss_dict = dict(
                valid_loss = 0,
                valid_loss_cmd = 0,
                valid_loss_args = 0
            )
            cnt = 0
            for data in tqdm.tqdm(dataloader):
                model_args = [data[arg].to(device) for arg in cfg.model_args]
                labels = data["label"].to(device) if "label" in data else None
                params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

                for i, loss_fn in enumerate(loss_fns, 1):
                    output = model(*model_args, params=params_dict)
                    loss_dict = loss_fn(output, labels, weights=weights_dict)
                cnt += 1
                
                valid_loss_dict['valid_loss'] += loss_dict['loss']
                valid_loss_dict['valid_loss_cmd'] += loss_dict['loss_cmd']
                valid_loss_dict['valid_loss_args'] += loss_dict['loss_args']
            
            valid_loss_dict['valid_loss'] /= cnt
            valid_loss_dict['valid_loss_cmd'] /= cnt
            valid_loss_dict['valid_loss_args'] /= cnt

            log = {
                **log,
                **valid_loss_dict,
            }
    
    return log


if __name__ == "__main__":
    # Reproducibility
    utils.set_seed(2022)
    # utils.init_global()

    parser = argparse.ArgumentParser(description='DeepSVG Trainer')
    parser.add_argument("--config-module", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--predict", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--desc", type=str, required=True)

    args = parser.parse_args()

    # when debugging, turn off wandb
    debug = args.debug
    predict = args.predict
    if debug or predict:
        os.environ["WANDB_MODE"] = "offline"

    cfg = importlib.import_module(args.config_module).Config(args.num_gpus)
    desc = args.desc

    train(cfg, desc, log_dir=args.log_dir, debug=debug, resume=args.resume, predict=args.predict)
