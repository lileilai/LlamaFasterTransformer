# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import multiprocessing
import numpy as np
from pathlib import Path
import torch 

import os
import sys
from transformers import GPTNeoXForCausalLM # 4.21.1

def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"

def quantize(mat, act_range):
    # qkv proj weight quantization
    if mat.ndim == 3 and mat.shape[1] == 3:
        # get max_q, max_k, max_v
        mat_max = np.abs(mat).clip(1e-8, None).max(axis=(0,2))[None, :, None]
    else:
        mat_max = np.abs(mat).clip(1e-8, None).max()

    act_scale_in = 127. / np.array(act_range[0])
    weight_scales = 127. / mat_max
    act_scale_post = 127. / np.array(act_range[1])

    mat_quant = (mat * weight_scales).round().astype(np.int8)
    return mat_quant, weight_scales, act_scale_in, act_scale_post

def split_and_convert_process(i, saved_dir, factor, key, old_name, args, config, capture_dict, val, dtype):


    def save_val(val, key, tp_num=None):
        path = saved_dir + "/model." + key
        if tp_num is not None:
            path += "." + str(tp_num)
        path += ".bin"

        val.tofile(path)


    quantized_out = args.act_scale is not None

    if key.find("input_layernorm.weight") != -1 or key.find("input_layernorm.bias") != -1 or \
        key.find("attention.dense.bias") != -1 or key.find("post_attention_layernorm.weight") != -1 or \
        key.find("post_attention_layernorm.bias") != -1 or key.find("mlp.dense_4h_to_h.bias") != -1 or \
        key.find("final_layernorm.weight") != -1 or key.find("final_layernorm.bias") != -1:

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            saved_path = saved_dir + "/model." + key + ".bin"
            val.tofile(saved_path)

    elif key.find("attention.dense.weight") != -1 or key.find("mlp.dense_4h_to_h.weight") != -1:
        
        if quantized_out:
            val_q, weight_scales, act_scale_pre, act_scale_post = quantize(val, capture_dict[old_name])
            save_val(act_scale_pre.astype(dtype), key.replace("weight", "scale"))
            scale_inter = (act_scale_post / (act_scale_pre * weight_scales)).astype(dtype)
            save_val(scale_inter, key.replace("weight", "scale_inter"))
            save_val((1. / act_scale_post).astype(dtype), key.replace("weight", "scale_out"))

            split_vals_q = np.split(val_q, factor, axis=0)
            for j in range(factor):
                save_val(split_vals_q[j], key + ".int8", i * factor + j)

        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif key.find("mlp.dense_h_to_4h.weight") != -1 or key.find("mlp.dense_h_to_4h.bias") != -1:
        

        if quantized_out and "weight" in key:
            val_q, weight_scales, act_scale_pre, act_scale_post = quantize(val, capture_dict[old_name])
            save_val(act_scale_pre.astype(dtype), key.replace("weight", "scale"))
            scale_inter = (act_scale_post / (act_scale_pre * weight_scales)).astype(dtype)
            save_val(scale_inter, key.replace("weight", "scale_inter"))
            save_val((1. / act_scale_post).astype(dtype), key.replace("weight", "scale_out"))

            split_vals_q = np.split(val_q, factor, axis=-1)
            for j in range(factor):
                save_val(split_vals_q[j], key + ".int8", i * factor + j)


        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.bias") != -1:
        local_dim = (int)(val.shape[-1] / 3)
        n_head = config['num_attention_heads']

        val = val.reshape(n_head, 3, local_dim // n_head)
        val = np.transpose(val, [1, 0, 2]).reshape(3, local_dim)
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    elif key.find("attention.query_key_value.weight") != -1:
        hidden_dim = val.shape[0]
        local_dim = (int)(val.shape[-1] / 3)
        n_head = config['num_attention_heads']
        # Note that the HF qkv weight are stored as [hidden_size, num_heads, 3, head_hidden]
        # FT needs the shape of [hidden_size, 3, num_heads, head_hidden]
        val = val.reshape(hidden_dim, n_head, 3, local_dim // n_head)
        val = np.transpose(val, [0, 2, 1, 3]).reshape(hidden_dim, 3, local_dim)

        if quantized_out:
            val_q, weight_scales, act_scale_pre, act_scale_post = quantize(val, capture_dict[old_name])
            weight_scales = weight_scales[0] * np.ones((3, local_dim // factor))
            # print(capture_dict[old_name])
            # print(act_scale_pre)
            # print(act_scale_post)

            act_scale_pre = np.array([act_scale_pre]).repeat(3)
            act_scale_post = np.array([act_scale_post]).repeat(3)

            # print(act_scale_pre[:, None].shape)

            save_val(act_scale_pre.astype(dtype), key.replace("weight", "scale"))
            scale_inter = (act_scale_post[:, None] / (act_scale_pre[:, None] * weight_scales)).astype(dtype)
            save_val(scale_inter, key.replace("weight", "scale_inter"))
            save_val((1. / act_scale_post).astype(dtype), key.replace("weight", "scale_out"))

            split_vals_q = np.split(val_q, factor, axis=-1)
            for j in range(factor):
                save_val(split_vals_q[j], key + ".int8", i * factor + j)
        # print(np.mean(np.abs(val[:, 0, :])))
        split_vals = np.split(val, factor, axis=-1)

        for j in range(factor):
            saved_path = saved_dir + "/model." + key + ".%d.bin" % (i * factor + j)
            split_vals[j].tofile(saved_path)

    else:
        print("[ERROR] cannot find key '{}'".format(key))

def split_and_convert(args):
    saved_dir = args.saved_dir + "/%d-gpu/" % args.infer_gpu_num

    if(os.path.exists(saved_dir) == False):
        os.makedirs(saved_dir)
    ckpt_name = args.in_file

    t_gpu_num = args.trained_gpu_num
    i_gpu_num = args.infer_gpu_num
    assert(i_gpu_num % t_gpu_num == 0)

    factor = (int)(i_gpu_num / t_gpu_num)

    # load position_embedding from rank 0 
    # model = torch.load(ckpt_name)
    model = GPTNeoXForCausalLM.from_pretrained(args.in_file)
    hf_config = vars(model.config)
    if "gpt_j_residual" not in hf_config:
        hf_config["gpt_j_residual"] = 0

    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    task_list = []
    if args.prompt_in_file_list is not None:
        task_list = prefix_prompt_convert(args, hf_config, np_weight_data_type)
    
    capture_dict = None
    if args.act_scale is not None:
        capture_dict = {}
        for key, values in torch.load(args.act_scale).items():
            capture_dict[key + ".weight"] = (values["input"], values["output"])


    try:
        model_name = args.model_name
        config = configparser.ConfigParser()
        config['gptneox'] = {}
        config['gptneox']['model_name'] = model_name
        config['gptneox']["head_num"] = str(hf_config["num_attention_heads"])
        n_embd = hf_config["hidden_size"]
        config['gptneox']["size_per_head"] = str(n_embd // hf_config["num_attention_heads"])
        config['gptneox']["inter_size"] = str(n_embd * 4)
        config['gptneox']["num_layer"] = str(hf_config["num_hidden_layers"])
        # rotary_dim = n_embd // hf_config["num_attention_heads"] if hf_config["rotary_dim"] is None else hf_config["rotary_dim"]

        rotary_dim = (n_embd // hf_config["num_attention_heads"]) * float(hf_config["rotary_pct"])

        config['gptneox']["rotary_embedding"] = str(rotary_dim)
        config['gptneox']["vocab_size"] = str(hf_config["vocab_size"])
        config['gptneox']["start_id"] = str(hf_config["bos_token_id"])
        config['gptneox']["end_id"] = str(hf_config["eos_token_id"])
        config['gptneox']['use_gptj_residual'] = str(1)
        config['gptneox']["weight_data_type"] = args.weight_data_type

        if len(task_list) > 0:
            config['gptneox']['num_tasks'] = str(len(task_list))
            config['gptneox']['prompt_learning_type'] = str(2)
            for idx, (task_name, prompt_length) in enumerate(task_list):
                config[f'task_{idx}'] = {}
                config[f'task_{idx}']['task_name'] = task_name
                config[f'task_{idx}']['prompt_length'] = str(prompt_length)
        with open((Path(saved_dir) / f"config.ini").as_posix(), 'w') as configfile:
            config.write(configfile)
    except:
        print(f"Fail to save the config in config.ini.")

    print("Finish save config.ini.")
    huggingface_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]
    
    ft_model_name_pattern = [
        "input_layernorm.bias",
        "input_layernorm.weight",
        "attention.query_key_value.bias",
        "attention.query_key_value.weight",
        "attention.dense.bias",
        "attention.dense.weight",
        "post_attention_layernorm.bias",
        "post_attention_layernorm.weight",
        "mlp.dense_h_to_4h.bias",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.bias",
        "mlp.dense_4h_to_h.weight",
    ]
    
    
    pool = multiprocessing.Pool(args.processes)
    for name, param in model.named_parameters():
        if name.find("weight") == -1 and name.find("bias") == -1:
            continue
        print(name)
        if name == 'gpt_neox.embed_in.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.wte.bin")
        elif name == 'gpt_neox.final_layer_norm.bias':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.final_layernorm.bias.bin")
        elif name == 'gpt_neox.final_layer_norm.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.final_layernorm.weight.bin")
        elif name == 'embed_out.weight':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.lm_head.weight.bin")
        elif name == 'embed_out.bias':
            param.detach().cpu().numpy().astype(np_weight_data_type).tofile(saved_dir + "model.lm_head.bias.bin")
        else:
            for i in range(len(huggingface_model_name_pattern)):
                if name.find(huggingface_model_name_pattern[i]) != -1:
                    new_name = name.replace("gpt_neox.", "").replace(huggingface_model_name_pattern[i], ft_model_name_pattern[i])
                    
                    pool.starmap(split_and_convert_process,
                                [(0, saved_dir, factor, new_name, name,
                                  args,
                                    vars(model.config),
                                    capture_dict,
                                    param.detach().cpu().numpy().astype(np_weight_data_type).T,
                                    np_weight_data_type
                                    )], 
                                )

    pool.close()
    pool.join()

    # Post-process biases if use_gptj_residual is True
    # if hf_config['gpt_j_residual']:
    if True:
        for layer_idx in range(hf_config["num_hidden_layers"]):
            attn_bias = np.fromfile(saved_dir + f"/model.layers.{layer_idx}.attention.dense.bias.bin", dtype=np.float32)
            mlp_bias =  np.fromfile(saved_dir + f"/model.layers.{layer_idx}.mlp.dense_4h_to_h.bias.bin", dtype=np.float32)

            (attn_bias + mlp_bias).tofile(saved_dir + f"/model.layers.{layer_idx}.mlp.attention.bias.sum.bin")

if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-saved_dir', '-o', type=str, help='file name of output file', required=True)
    parser.add_argument('-in_file', '-i', type=str, help='file name of input checkpoint file', required=True)
    parser.add_argument('-prompt_in_file_list','-p_i_list', type=str, help='list of the prompt weight file path,'
                        'separate by (,). e.g. -prompt_in_file_list prefix_prompt.task0.weight,prefix_prompt.task1.weight')
    parser.add_argument('-trained_gpu_num', '-t_g', type=int, help='How many gpus for inference', default=1)
    parser.add_argument('-infer_gpu_num', '-i_g', type=int, help='How many gpus for inference', required=True)
    parser.add_argument("-processes", "-p", type=int, help="How many processes to spawn for conversion (default: 4)", default=4)
    parser.add_argument("-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument('-model_name', '-m_n', type=str, help='model name', required=True)
    parser.add_argument("-act_scale", default=None, help="path to activation scalings for int8 conversion")


    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("========================================")

    split_and_convert(args)