import torch
import clip
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPImageQualityAssessment
import edit_distance
import transformers
from tqdm import trange
from modelzipper.tutils import *
import gc


@torch.no_grad()
def calculate_fid(fid_metric, pred_images, golden_images, clip_model, clip_process, device):
    """
    a single tensor it should have shape (N, C, H, W). If a list of tensors, each tensor should have shape (C, H, W). C is the number of channels, H and W are the height and width of the image.
    
        imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)  
        imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
        fid.update(imgs_dist1, real=True)
        fid.update(imgs_dist2, real=False)
    """
    pred_image_features, golden_image_features = [], []
    for i in trange(len(pred_images)):
        pred_image = clip_process(Image.open(pred_images[i])).to(device)
        golden_image = clip_process(Image.open(golden_images[i])).to(device)
        pred_image_features.append(pred_image)
        golden_image_features.append(golden_image)
    
    pred_images = torch.stack(pred_image_features, dim=0).to(dtype=torch.uint8)
    golden_images = torch.stack(golden_image_features, dim=0).to(dtype=torch.uint8)
    
    fid_metric.update(golden_images, real=True)  # N x C x W x H
    fid_metric.update(pred_images, real=False)
    return fid_metric.compute()

@torch.no_grad()
def calculate_clip_core(clip_process, clip_metric, pred_images, keywords_lst):
    """
        metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
        score = metric(torch.randint(255, (3, 224, 224), generator=torch.manual_seed(42)), "a photo of a cat")
        score.detach()
    """
    avg_scores = []
    for i in trange(len(pred_images)):
        img = clip_process(Image.open(pred_images[i])).unsqueeze(0).to(device).to(dtype=torch.uint8)
        clip_score = clip_metric(img, keywords_lst[i])
        avg_scores.append(clip_score)
        
    return sum(avg_scores) / len(avg_scores)


def calculate_clip_image_quality(quality_metric, imgs):
    """
        _ = torch.manual_seed(42)
        imgs = torch.randint(255, (2, 3, 224, 224)).float()
        metric = CLIPImageQualityAssessment(prompts=("quality"))
        metric(imgs)
    """
    return quality_metric(imgs)


def calculate_edit(tokenizer, gen_svg_paths, golden_svg_paths):
    preds = [tokenizer.tokenize(x) for x in gen_svg_paths]     
    avg_str_prd_len = sum([len(x) for x in preds]) / len(preds)
    golden = [tokenizer.tokenize(x) for x in golden_svg_paths]      
    distance = []
    for i in trange(len(preds)):
        sm = edit_distance.SequenceMatcher(a=preds[i], b=golden[i])
        distance.append(sm.distance())
    return sum(distance) / len(distance), avg_str_prd_len


def calculate_hps(image_lst1, image_lst2, key_lst, clip_model, clip_process,):
   
    image1 = clip_process(Image.open("image1.png")).unsqueeze(0).to(device)
    image2 = clip_process(Image.open("image2.png")).unsqueeze(0).to(device)
    images = torch.cat([image1, image2], dim=0)
    text = clip.tokenize(["your prompt here"]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(images)
        text_features = clip_model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        hps = image_features @ text_features.T


if __name__ == "__main__":
    
    ## text VQ 
    FILE_PATH = "/zecheng2/evaluation/test_vq/version_8/vq_test.pkl"
    data = auto_read_data(FILE_PATH)
    
    keys = [item['keys'] for item in data]
    pi_res_len = [item['pi_res_len'] for item in data]
    pc_res_len = [item['pc_res_len'] for item in data]
    gt_res_len = [item['gt_res_len'] for item in data]
    pi_res_str = [item['pi_res_str'] for item in data]
    pc_res_str = [item['pc_res_str'] for item in data]
    gt_str = [item['gt_str'] for item in data]
    PI_RES_image_path = [item['PI_RES_image_path'] for item in data]
    PC_RES_image_path = [item['PC_RES_image_path'] for item in data]
    GT_image_path = [item['GT_image_path'] for item in data]
    
    # dict_keys(['text', 'p_svg_str', 'g_svg_str', 'r_svg_str', 'r_svg_path', 'p_svg_path', 'g_svg_path'])
    
    device =  "cuda:1"
    fid_metric = FrechetInceptionDistance(feature=768).to(device)  # 768
    clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
   
    quality_metric = CLIPImageQualityAssessment(prompts=("quality",))
    
    clip_model, clip_process = clip.load("ViT-L/14", device=device)
    
    # pred_images = [item['p_svg_path'] for item in data]
    # reconstruction_images = [item['r_svg_path'] for item in data]
    # golden_images = [item['g_svg_path'] for item in data]
    metrics = {}
    
    metrics['pi_res_len'] = pi_res_len
    metrics['pc_res_len'] = pc_res_len
    metrics['gt_res_len'] = gt_res_len
    
    PI_fid_res = calculate_fid(fid_metric, PI_RES_image_path, GT_image_path, clip_model, clip_process, device).cpu()
    PC_fid_res = calculate_fid(fid_metric, PC_RES_image_path, GT_image_path, clip_model, clip_process, device).cpu()
    
    metrics['PI_fid_res'] = PI_fid_res
    metrics['PC_fid_res'] = PC_fid_res
    
    
    PI_CLIP_SCORE = calculate_clip_core(clip_process, clip_metric, PI_RES_image_path, keys)
    PC_CLIP_SCORE = calculate_clip_core(clip_process, clip_metric, PC_RES_image_path, keys)
    
    metrics['PI_CLIP_SCORE'] = PI_CLIP_SCORE
    metrics['PC_CLIP_SCORE'] = PC_CLIP_SCORE
    
    # text metrics
    t5_tokenizer = transformers.T5Tokenizer.from_pretrained("/zecheng2/model_hub/flan-t5-xl")
    edit_score_pi, pi_str_len = calculate_edit(t5_tokenizer, pi_res_str, gt_str)
    edit_score_pc, pc_str_len = calculate_edit(t5_tokenizer, pc_res_str, gt_str)
    
    metrics['edit_score_pi'] = edit_score_pi
    metrics['edit_score_pc'] = edit_score_pc
    metrics['pi_str_len'] = pi_str_len
    metrics['pc_str_len'] = pc_str_len
    
    print(metrics)
    
    exit()
    clip_model2, _ = clip.load("ViT-L/14", device=device)
    params = torch.load("/zecheng2/evaluation/hpc.pt")['state_dict']
    clip_model2.load_state_dict(params)
    
    import pdb; pdb.set_trace()
    hps_score = calculate_hps(image_lst1, image_lst2, key_lst)

