# custom numerical llama

import sys
import json
import re 
import transformers
import random
import torch  
import torch.nn as nn 
from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, List, Dict, Any
from tqdm import tqdm  
import torch.nn.functional as F
from typing import Any, Mapping, Tuple, List
from transformers import PreTrainedTokenizer, LlamaConfig, LlamaForCausalLM  
from torch import Tensor 
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from typing import Optional, Dict, Sequence
from vector_quantize_pytorch import ResidualVQ
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from torch_geometric.nn.conv import SAGEConv
from torchtyping import TensorType
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates
from local_attention import LocalMHA

def top_p(scores, p, temperature):
    scores = scores / temperature
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # pdb.set_trace()
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -1 :] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, -float("Inf"))
    return scores


# tensor helper functions

@beartype
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

@beartype
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo


def extract_edge_ids_from_svg(svg_path, num_sub_command = 3, pad_id=-1):
    device = svg_path.device
    edge_id_pairs, batch_edge_id_pairs = [], []
    current_command_id = 0
    if len(svg_path) < num_sub_command * 2:
        return edge_id_pairs
    for batch_data in svg_path:
        for i in range(0, len(batch_data)):
            if batch_data[i][0] != -1:
                if i + num_sub_command < len(batch_data):
                    if batch_data[i + num_sub_command][0] != -1:
                        edge_id_pairs.append((current_command_id, current_command_id + 1))
                        current_command_id += 1
            else:
                batch_edge_id_pairs.append(edge_id_pairs)
                break

    ## add padding to edge_id_pairs
    max_len = max([len(item) for item in batch_edge_id_pairs])
    for item in batch_edge_id_pairs:
        item.extend([(pad_id, pad_id)] * (max_len - len(item)))

    return torch.tensor(batch_edge_id_pairs, dtype=torch.long, device=device)


class SVGAutoencoder(nn.Module):

    def __init__(
            self, 
            conv_dim = 512, 
            sageconv_kwargs: dict = dict(
                normalize = True,
                project = True
            ),
            num_discrete_coors = 200, 
            num_commands = 3, 
            encoder_depth = 2,
            decoder_depth = 2,
            dim_coor_embed = 4096, 
            backbone_dim = 4096,
            final_encoder_norm = True,
            sageconv_dropout = 0.,
            attn_dropout = 0.,
            ff_dropout = 0.,
            resnet_dropout = 0.,
            local_attn_window_size = 128,
            local_attn_encoder_depth = 0,
            local_attn_decoder_depth = 0,
            local_attn_kwargs: dict = dict(
                dim_head = 32,
                heads = 8
            ),
        ):
        super().__init__()
        self.num_discrete_coors = num_discrete_coors
        self.type_embed = nn.Embedding(num_commands, dim_coor_embed)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # project into model dimension
        self.project_in = nn.Linear(dim_coor_embed, backbone_dim)

        # encoder
        self.encoder = nn.ModuleList([])

        for _ in range(encoder_depth):
            sage_conv = SAGEConv(
                conv_dim,
                conv_dim,
                sageconv_dropout = sageconv_dropout,
                **sageconv_kwargs
            )

            self.encoders.append(sage_conv)

        self.final_encoder_norm = nn.LayerNorm(backbone_dim) if final_encoder_norm else nn.Identity()


        # local attention related

        self.encoder_local_attn_blocks = torch.nn.ModuleList([])
        self.decoder_local_attn_blocks = torch.nn.ModuleList([])

        attn_kwargs = dict(
            dim = backbone_dim,
            causal = False,
            prenorm = True,
            dropout = attn_dropout,
            window_size = local_attn_window_size,
        )

        for _ in range(local_attn_encoder_depth):
            self.encoder_local_attn_blocks.append(nn.ModuleList([
                LocalMHA(**attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(backbone_dim), FeedForward(backbone_dim, glu = True, dropout = ff_dropout))
            ]))

        for _ in range(local_attn_decoder_depth):
            self.decoder_local_attn_blocks.append(nn.ModuleList([
                LocalMHA(**attn_kwargs, **local_attn_kwargs),
                nn.Sequential(RMSNorm(backbone_dim), FeedForward(backbone_dim, glu = True, dropout = ff_dropout))
            ]))


    @beartype
    def quantize(
        self,
        *,
        faces: TensorType['b', 'nf', 3, int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, num_faces, device = *faces.shape[:2], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nv d) -> b nf nv d', nv = 3)

        vertex_dim = face_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim = -2, value = 0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nv -> b (nf nv) d', d = vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict()

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # residual VQ

        quantized, codes, commit_loss = self.quantizer(averaged_vertices, mask = mask, **quantize_kwargs)

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = quantized.gather(-2, faces_with_dim)
        face_embed_output = rearrange(face_embed_output, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        face_embed_output = self.project_codebook_out(face_embed_output)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        faces_with_quantized_dim = repeat(faces, 'b nf nv -> b (nf nv) q', q = self.num_quantizers)
        codes_output = codes.gather(-2, faces_with_quantized_dim)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nv) 1', nv = 3)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss

    @beartype
    def encode(
            self, 
            svg_commands: TensorType['b', 'nc', 9, float],
            svg_paths:    TensorType['b', 'np', 3, int],
            svg_edges:    TensorType['b', 'e', 50, int],
            svg_mask:     TensorType['b', 'np', bool],
            svg_edge_mask:TensorType['b', 'e', bool],
        ):
        """
            einops:
            b - batch
            ns - number of svg paths
            nc - number of commands (9)
            c - commands (9)
            d - embed dim
        """

        batch_size, num_sub_paths, num_commands = svg_paths.size()
        _, num_paths, _ = svg_edges.shape

        path_without_pad = svg_paths.masked_fill(~rearrange(svg_mask, 'b np -> b np 1'), 0)

        path_ = repeat(path_without_pad, 'b np c -> b np c d', d = num_commands)
        svg_commands = repeat(svg_commands, 'b nc c -> b np nc c', np = num_paths)

        path_commands = svg_commands.gather(-2, path_) # get all path commands

        type_embed = self.type_embed(path_commands[..., 0])
        coord_embed = self.coor_embed(path_commands[..., 1:])

        path_embed = torch.cat((type_embed, coord_embed), dim = -1)
        path_embed = rearrange(path_embed, 'b np nc d -> b np (nc d)')

        # project into model dimension
        path_embed = self.project_in(path_embed)
        orig_path_embed_shape = path_embed.shape

        svg_edges = svg_edges[svg_edge_mask]
        svg_edges = rearrange(svg_edges, 'be ij -> ij be')

        for conv in self.encoders:
            path_embed = conv(path_embed, svg_edges)

        path_embed = path_embed.new_zeros(orig_path_embed_shape).masked_scatter(rearrange(svg_mask, '... -> ... 1'), path_embed)

        for attn, ff in self.encoder_local_attn_blocks:
            path_embed = attn(path_embed, mask = svg_mask) + path_embed
            path_embed = ff(path_embed) + path_embed

        path_embed = self.final_encoder_norm(path_embed)

        return path_embed, path_commands


    def forward(
        self,
        svg_commands: TensorType['b', 'nc', 9, float],
        svg_paths:    TensorType['b', 'np', 3, int],
        svg_edges:    TensorType['b', 'e', 50, int] = None,
        rvq_sample_codebook_temp = 1,
    ):

        if svg_edges is None:
            svg_edges = extract_edge_ids_from_svg(svg_paths, pad_id=-1)  # get edges from commands directly
        
        svg_mask = reduce(svg_paths != self.pad_id, 'b nf c -> b nf', 'all')
        svg_edges_mask = reduce(svg_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, svg_coord = self.encode(svg_commands, svg_paths, svg_edges, svg_mask, svg_edges_mask)





class NumericalSVGDataset(Dataset):
    
    def __init__(self, args, svg_file, tokenizer: PreTrainedTokenizer, numerical_token):  
        self.tokenizer = tokenizer  
        self.max_seq_len = args.model_max_length
        self.numerical_token = numerical_token
        self.numerical_token_id = self.tokenizer.convert_tokens_to_ids(self.numerical_token)
        
        ## Load SVG data
        with open(svg_file, "r") as f2:
            self.content = [json.loads(line) for line in f2]
        
        ## whether to process numerical values in svg paths
        self.numerical_mode = True
        if args.numerical_mode is not None:
            self.numerical_mode = args.numerical_mode
        
    def __len__(self):
        return len(self.content)
    
    def extract_numerical(self, path_data, numerical_token):  
        ## match all numericals 
        number_pattern = r"-?\d+\.?\d*"  
        numbers = re.findall(number_pattern, path_data)  
        numbers = [float(item) for item in numbers]
        ## replace all matched numericals with numerical_token ("[NUM]")  
        replaced_data = re.sub(number_pattern, numerical_token, path_data)
        replaced_data = re.sub(' +', ' ', replaced_data)
        replaced_data = re.sub(r'([MLC]) ', r'\1', replaced_data) 
        replaced_data = re.sub(r'(\[NUM\]) ', r'\1', replaced_data) 
        # import pdb; pdb.set_trace()
         
        # pattern = re.compile(r'(?<=\[NUM\])\s+(?=\[NUM\])')  
        # replaced_data = pattern.sub('', replaced_data)
        # pattern2 = re.compile(r'(?<=\b[LCM])\s+(?=\[NUM\])')
        # replaced_data = pattern2.sub('', replaced_data)
        # replaced_data = re.sub(r'(?<=[CML])\s+|\s+(?=\[NUM\])', '', replaced_data)
        return numbers, replaced_data 
    
    def extract_c_segments(self, path_data):  
        ## find all the control path in svg paths
        c_pattern = r"c[^A-Za-z]*?(?=[A-Za-z])"  
        c_segments = re.findall(c_pattern, path_data)  
        if len(c_segments) == 1:  # only one control path, usually complex
            return [self.extract_consecutive_numbers(c_segments[0], 0.5)]
        return c_segments 
    
    def __getitem__(self, index):
        data = self.content[index]
        # svg_path = data["compress_path"].split("#Begin:")[-1].strip()
        svg_path_with_prompt = data["compress_path"]
        
        ## Step 1: extract numericals from svg paths 
        ## and replace the numericals with numerical_token
        extracted_numericals, replaced_paths = self.extract_numerical(
            svg_path_with_prompt, self.numerical_token)
        
        ## Step 2: encode the replaced svg paths
        if self.numerical_mode:
            seq_inputs = self.tokenizer(
                replaced_paths, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            text_input_ids = seq_inputs.input_ids[0]
            text_attention_mask = seq_inputs.attention_mask[0]
            text_labels = torch.where(
                text_input_ids != self.tokenizer.pad_token_id, text_input_ids, -100
            )

        return {
            "input_ids": text_input_ids,
            "attention_mask": text_attention_mask,
            "labels": text_labels,
            "numerical_values": extracted_numericals,
        }
        
    @classmethod
    def custom_datacollator(cls, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate examples for supervised fine-tuning."""
        
        batch_input_ids, batch_attn_mask, batch_label = [], [], []
        batch_numerical_input_ids = []
        max_numerical_nums = max([len(item["numerical_values"]) for item in instances])
        
        for ins in instances:
            batch_input_ids.append(ins["input_ids"])
            batch_attn_mask.append(ins["attention_mask"])
            batch_label.append(ins["labels"])
            
            ## process numerical values
            ### Step 1: convert to float tensor
            numerical_values = torch.FloatTensor(ins["numerical_values"])
            ### Step 2: pad to the same length
            numerical_values = torch.cat(  
                [numerical_values, torch.full((max_numerical_nums - len(numerical_values),), 255)]  # use 255 for padding
            )  

            batch_numerical_input_ids.append(numerical_values)
            
        batch_input_ids = torch.stack(batch_input_ids, dim=0)
        batch_attn_mask = torch.stack(batch_attn_mask, dim=0)
        batch_label = torch.stack(batch_label, dim=0)
        batch_numerical_input_ids = torch.stack(batch_numerical_input_ids, dim=0)
        
        return {
            "batch_input_ids": batch_input_ids,
            "batch_numerical_input_ids": batch_numerical_input_ids,
            "batch_attention_mask": batch_attn_mask,
            "batch_labels": batch_label,
        }


class NumericalProcessor(nn.Module):
    def __init__(self, backbone_dim=4096, num_bins=8, hidden_dims=None, encoding_method="binary") -> None:
        super(NumericalProcessor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [num_bins, 64, 512, backbone_dim]
            
        self.encoding_method = encoding_method
        self.num_bins = num_bins
        
        modules = [] # build up-projector
        in_dim = num_bins
        for h_dim in hidden_dims:    
            modules.append(    
                nn.Sequential(    
                    nn.Linear(    
                        in_features=in_dim, out_features=h_dim    
                    ),    
                    nn.LayerNorm(h_dim),    
                    nn.Sigmoid()    ## FIXME: use PekReLU or Tanh? follow llama model
                )    
            )    
            in_dim = h_dim    
        self.encoder = nn.Sequential(*modules)
        
        
        assert backbone_dim % num_bins == 0, "backbone_dim must be divisible by num_bins"
        downProject_dim = backbone_dim // num_bins
        
        self.decode_projector = nn.Linear(backbone_dim, backbone_dim)
        self.classifier = nn.Linear(downProject_dim, 2)
       
    def int_to_binary_list(self, num):  
        binary_str = bin(num)[2:]  
        return list(map(int, binary_str))  

    
    def long_to_binary_tensor(self, tensor, num_bits):  
        binary_tensor = torch.zeros(tensor.size() + (num_bits,), dtype=torch.long, device=tensor.device)  
        for i in range(num_bits):  
            binary_tensor[..., i] = torch.bitwise_and(tensor >> i, 1)  
        return binary_tensor  
    
    def binary_to_long(self, binary_tensor, num_bits):  
        long_tensor = torch.zeros(binary_tensor.shape[:-1], dtype=torch.long, device=binary_tensor.device)  
        for i in range(num_bits):   
            # Shift the bit i places to the left and combine it with the current long tensor  
            long_tensor += (binary_tensor[..., i] << i).type(torch.long)  
        return long_tensor
    
    def encode(self, seq: torch.LongTensor) -> torch.FloatTensor:
        """
        seq: b x l
        """
        if self.encoding_method == "binary":
            original_tensor_type = seq.dtype
            seq = seq.long()
            encoded_seq = self.long_to_binary_tensor(seq, self.num_bins)  # b x l x num_bins
            encoded_seq = encoded_seq.to(original_tensor_type)
            encoded_hidden_states = self.encoder(encoded_seq)  # b x l x backbone_dim
        
        elif self.encoding_method == "log":
            pass
        
        return encoded_seq, encoded_hidden_states


    def decode(self, hidden_states: torch.FloatTensor) -> torch.LongTensor:
        """
        hidden_states: b x l x backbone_dim
        """
        if self.encoding_method == "binary":
            decoded_res = self.decoder(hidden_states)  # b x l x num_bins
        elif self.encoding_method == "log":
            pass
        
        return decoded_res
    


def _init_quantizier(vq_name, vq_config):
    if vq_name == "residualVQ":
        return ResidualVQ(
            dim = vq_config['dim'],
            num_quantizers = vq_config['num_quantizers'],
            codebook_size = vq_config['codebook_size'],
        )

    else:
        raise NotImplementedError(f"VQ {vq_name} not implemented")


class VQLLaMA(LlamaForCausalLM, GenerationMixin):  
    def __init__(self, config, quantizer_loss_weight=2.0, tokenizer=None, hidden_dim=None, encoding_method="residualVQ", vq_config=None, num_bins=9):  
        super(VQLLaMA, self).__init__(config)
        self.up_projector = nn.Linear(num_bins, hidden_dim)
        self.down_projector = nn.Linear(hidden_dim, num_bins)
        self.quantizer = _init_quantizier(encoding_method, vq_config)
        self.quantizer_loss_weight = quantizer_loss_weight
        self.tokenizer = tokenizer
        self.post_init()
        
        if config.frozen_llm: 
            print("Attention! Part of the parameters are freezed!")
            self.requires_grad_ = False 
            self.svg_embedding.requires_grad_ = True
            self.svg_lm_head.requires_grad_ = True
    

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict)
    
    @torch.no_grad()
    def nucle_sampling_generate(self, input_ids=None, past_key_values=None, max_length=None, min_length=None, **kwargs):
        
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        last_hidden_state = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        text_logits = self.lm_head(last_hidden_state).float()
        next_token_scores = top_p(text_logits[:, -1, :], 0.9, 0.6)
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        pred_token_idx = torch.multinomial(probs, num_samples=1)
        generated_ids = [pred_token_idx.item()]
        pos = 0
        
        for _ in range(max_length - 1):
            outputs = self.model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            text_logits = self.lm_head(last_hidden_state).float()
            next_token_scores = top_p(text_logits[:, -1, :], 0.9, 0.6)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            pred_token_idx = torch.multinomial(probs, num_samples=1)
            generated_ids.append(pred_token_idx.item())

            if pred_token_idx == self.tokenizer.eos_token_id:
                break

        return generated_ids
    
    
    @torch.no_grad()
    def greedy_generate(self, input_ids=None, past_key_values=None, max_length=None, min_length=None, **kwargs):
        
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        last_hidden_state = outputs.last_hidden_state
        past_key_values = outputs.past_key_values
        text_logits = self.lm_head(last_hidden_state).float()
        
        pred_token_idx = text_logits[:, -1, :].argmax(dim=-1).unsqueeze(1) # 1x1
        generated_ids = [pred_token_idx.item()]
        pos = 0
        
        for _ in range(max_length - 1):
            outputs = self.model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            text_logits = self.lm_head(last_hidden_state).float()
            pred_token_idx = text_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_ids.append(pred_token_idx.item())
            
            if pred_token_idx == self.tokenizer.eos_token_id:
                break
            
        return generated_ids
    
    
    @torch.no_grad()
    def adaptive_generate(self, input_ids=None, past_key_values=None, max_length=None, min_length=None, **kwargs):
        
        m_token_id = 29924
        l_token_id = 29931
        c_token_id = 29907
        num_token_id = self.tokenizer.encode(self.numerical_token)[1]
        
        def inner_loop(input_ids, past_key_values):
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            text_logits = self.lm_head(last_hidden_state).float()
            return text_logits, past_key_values
        
        text_logits, past_key_values = inner_loop(input_ids, past_key_values)
        pred_token_idx = text_logits[:, -1, :].argmax(dim=-1).unsqueeze(1) # 1x1
        generated_ids = [pred_token_idx.item()]
        generated_nums = []
        num_cnt = 0
        numerical_embeddings = None
        predict_next_tag = False
        boost_value = 5 
        
        for _ in range(max_length - 1):
            if numerical_embeddings is not None:
                outputs = self.model(
                    inputs_embeds=numerical_embeddings,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = self.model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            decoded_num = None
            if num_cnt != 0:
                projection_res = self.numerical_processer.decode_projector(last_hidden_state)
                shaped_res = projection_res.view(last_hidden_state.size(0), 1, self.num_bins, -1)
                cls_results = self.numerical_processer.classifier(shaped_res)
                binary_res = cls_results.argmax(-1).squeeze()
                decoded_num = self.numerical_processer.binary_to_long(binary_res, 8)
                num_cnt -= 1            
            
            text_logits = self.lm_head(last_hidden_state).float()
            
            if predict_next_tag:
                text_logits[0, 0, m_token_id] += boost_value  
                text_logits[0, 0, l_token_id] += boost_value  
                text_logits[0, 0, c_token_id] += boost_value 
                softmax_logits = torch.nn.functional.softmax(text_logits[0, 0], dim=-1)  
                pred_token_idx = torch.argmax(softmax_logits).unsqueeze(0).unsqueeze(0) 
            else:
                pred_token_idx = text_logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            
            # import pdb; pdb.set_trace()
            
            if num_cnt == 0 and pred_token_idx in (m_token_id, c_token_id, l_token_id):
                if pred_token_idx == m_token_id or pred_token_idx == l_token_id:
                    num_cnt = 2
                elif pred_token_idx == c_token_id:
                    num_cnt = 6
                predict_next_tag = True
                
            if decoded_num is not None:
                generated_ids.append(num_token_id)
                generated_nums.append(decoded_num)
            else:
                generated_ids.append(pred_token_idx.item())
            
            if pred_token_idx == self.tokenizer.eos_token_id:
                break
            
        return generated_ids, generated_nums
    
    
    def forward(self, text_inputs=None, numerical_inputs=None, labels=None, attention_masks=None, **kwargs): 
        """
        input_ids: {
            "text_inputs": b x l, 
            "numerical_inputs": b x l,
            "batch_attn_mask": b x l,
            "batch_labels": b x l,
        }
        """
        
        ## Encode textual information
        text_embedding_module = self.base_model.get_input_embeddings()
        input_embeddings = text_embedding_module(text_inputs)
        
        ## Encode numerical information
        processed_num, numerical_embeddings = self.numerical_processer.encode(numerical_inputs)
        batch_size, seq_len = text_inputs.size()
        numerical_mask = text_inputs == self.numerical_token_id
        numerical_counts = numerical_mask.sum(dim=1).tolist() 
        encoded_numerical_h = [item[:count] for item, count in zip(numerical_embeddings, numerical_counts)]  # List[Tensor]

        ## construct the golden numerical
        golden_numericals = [item[:count, :] for item, count in zip(processed_num, numerical_counts)]
        
        ## Replace all the numerical positions in text embeddings with numerical embeddings 
        for i in range(batch_size):
            numerical_index = 0
            for j in range(seq_len):
                if text_inputs[i, j] == self.numerical_token_id:
                    input_embeddings[i, j, :] = encoded_numerical_h[i][numerical_index, :]
                    numerical_index += 1
                if text_inputs[i, j] == self.tokenizer.eos_token_id:
                    assert encoded_numerical_h[i].size(0) == numerical_index, "Numerical index not match"
                    break
                
        ## pass the embeddings to backbone model        
        outputs = self.model(
            input_ids=None, 
            attention_mask=attention_masks,
            inputs_embeds=input_embeddings, 
            **kwargs
        )
        hidden_states = outputs[0]
        
        # Decode numerical information
        ## extract all the numerical embeddings
        numerical_res = []
        for i in range(batch_size):
            minibatch_numerical = []
            for j in range(seq_len):
                if text_inputs[i, j] == self.numerical_token_id:
                    minibatch_numerical.append(hidden_states[i, j-1, :].unsqueeze(0))
                if text_inputs[i, j] == self.tokenizer.eos_token_id:
                    break
            numerical_res.append(torch.cat(minibatch_numerical, dim=0)) # List[List[Tensor]]

        ## decode numerical embeddings with down projector
        decoded_results = [self.numerical_processer.decode_projector(z.unsqueeze(0)) for z in numerical_res]  # List[Tensor]
        shaped_results = [item.view(item.size(0), item.size(1), self.num_bins, -1) for item in decoded_results]  # List[Tensor]
        cls_results = [self.numerical_processer.classifier(item) for item in shaped_results]  # List[Tensor]
        
        loss_fct = CrossEntropyLoss()
        
        ## calculate the cls Loss
        recon_loss = 0
        for i in range(batch_size):
            golden_numerical = golden_numericals[i].squeeze(0).long()  # 1 x seq_len x num_bins
            decode_result = cls_results[i] # 1 x seq_len x num_bins
            
            loss = loss_fct(decode_result.view(-1, 2), golden_numerical.view(-1))
            recon_loss += loss
            # assert golden_numerical.size() == decode_result.size(), "Numerical input size not match"
            # recon_loss += torch.sum(F.mse_loss(decode_result, golden_numerical, reduction="none"), dim=1).mean() 
        
        recon_loss = recon_loss / batch_size
        
        ## Decode textual information
        text_logits = self.lm_head(hidden_states).float()

        ### Step 2: calculate the LM Loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = text_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # EMU Loss
            emu_shift_labels = torch.where(shift_labels == self.numerical_token_id, 
                           torch.tensor(-100, device=shift_logits.device), 
                           shift_labels)

            loss = loss_fct(shift_logits, emu_shift_labels)

        total_loss = loss + self.numerical_loss_weight * recon_loss
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "textual_loss": loss,
            "logits": text_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, max_length=None, min_length=None, **kwargs):
        
        return super().generate(input_ids, attention_mask, max_length, min_length, **kwargs)
    
    
    @property
    def numerical_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.numerical_token)
    
    @property
    def model_device(self):
        return next(self.parameters()).device

   
if __name__ == "__main__":

    ## fake data for testing VQVAE
    fake_data = torch.randint(0, 200, (64, 9))
    fake_data[:, 0] = torch.randint(0, 3, (64,))
    mask = (fake_data[:, 0] == 0) | (fake_data[:, 0] == 1)
    fake_data[mask, 3:7] = 0

    autoencoder = SVGAutoencoder()

    res = autoencoder(fake_data)

    exit()
    
    model_name_or_path = "/zecheng/svg_model_hub/NLLaMA-V3-emu-nospace/checkpoint-700"
    DEFAULT_PAD_TOKEN = "[PAD]"
    NUMERICAL_TOKEN = "[NUM]"
    
    # Step 1: Load Model Config 
    llamaconfig = transformers.LlamaConfig.from_pretrained(model_name_or_path)
    llamaconfig.frozen_llm = False
    
    # Step 2: Load Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=1024,
        padding_side="left"
    )
    tokenizer.pad_token_id = tokenizer.unk_token_id
    
    # Step 2: Load Model (without tokenizer)
    model = NumericalLlama.from_pretrained(
        model_name_or_path, 
        config=llamaconfig, 
        numerical_token=NUMERICAL_TOKEN,
        hidden_dim=llamaconfig.hidden_size,
        num_bins=8,
        device_map={"":0},
        tokenizer=tokenizer,
    )
    model.config.pad_token_id = 0
    
    prompt = "Keywords: speaker #Begin:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda:0')
    
    generated_ids, generated_nums = model.adaptive_generate(input_ids=input_ids, max_length=1024)
    # res = model.greedy_generate(input_ids=input_ids, max_length=1024)
    generated_text = tokenizer.decode(generated_ids)
    import pdb; pdb.set_trace()
    print(generated_text)
    print(generated_nums)