/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/llama/LlamaDecoderLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(const int  hidden_units,
                                                        const int  inter_size,
                                                        const int  tensor_para_size,
                                                        const int  tensor_para_rank,
                                                        const bool use_gptj_residual):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    use_gptj_residual_(use_gptj_residual)
{
    mallocWeights();
    setWeightPtr();
}

template<typename T>
LlamaDecoderLayerWeight<T>::~LlamaDecoderLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 14; i++) {
            if (!use_gptj_residual_ && i != attention_dense_bias_weight_id) {
                cudaFree(weights_ptr[i]);
            }
        }

        pre_layernorm_weights.beta                            = nullptr;
        pre_layernorm_weights.gamma                           = nullptr;
        self_attention_weights.query_weight.kernel            = nullptr;
        self_attention_weights.query_weight.bias              = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias   = nullptr;
        post_attention_layernorm_weights.beta                 = nullptr;
        post_attention_layernorm_weights.gamma                = nullptr;

        ffn_weights.intermediate_weight.kernel  = nullptr;
        ffn_weights.intermediate_weight2.kernel = nullptr;
        ffn_weights.output_weight.kernel        = nullptr;
        
        ffn_weights.intermediate_weight.bias  = nullptr;
        ffn_weights.intermediate_weight2.bias = nullptr;
        ffn_weights.output_weight.bias        = nullptr;

        is_maintain_buffer                     = false;
    }
}

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(const LlamaDecoderLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    use_gptj_residual_(other.use_gptj_residual_)
{
    mallocWeights();

    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    if (!use_gptj_residual_) {
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    }

    

    // gate_weight & bias
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);


    // up_weight & bias
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);

    // down_weight & bias
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    // post_layer_norm
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);
    setWeightPtr();
}

template<typename T>
LlamaDecoderLayerWeight<T>& LlamaDecoderLayerWeight<T>::operator=(const LlamaDecoderLayerWeight& other)
{
    hidden_units_      = other.hidden_units_;
    inter_size_        = other.inter_size_;
    tensor_para_size_  = other.tensor_para_size_;
    tensor_para_rank_  = other.tensor_para_rank_;
    use_gptj_residual_ = other.use_gptj_residual_;

    mallocWeights();

    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    if (!use_gptj_residual_) {
        cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    }

    // gate_proj
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);

    // up_proj
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[9], other.weights_ptr[9], inter_size_ / tensor_para_size_);

    // down_proj
    cudaD2Dcpy(weights_ptr[10], other.weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[11], other.weights_ptr[11], hidden_units_);

    // post_layer_norm
    cudaD2Dcpy(weights_ptr[12], other.weights_ptr[12], hidden_units_);
    cudaD2Dcpy(weights_ptr[13], other.weights_ptr[13], hidden_units_);

    setWeightPtr();
    return *this;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);
    const std::string rank_spec = std::to_string(tensor_para_rank_);

    loadWeightFromBin<T>(
        weights_ptr[0], {(size_t)hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[1], {(size_t)hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {(size_t)hidden_units_, (size_t)(3 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.weight." + rank_spec + ".bin",
                         model_file_type);

    const DataType data_type = getTensorType<T>();

    loadWeightFromBin<T>(weights_ptr[3],
                         {(size_t)(3 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query_key_value.bias." + rank_spec + ".bin",
                         model_file_type);

    

    loadWeightFromBin<T>(weights_ptr[4],
                         {(size_t)(hidden_units_ / tensor_para_size_), (size_t)hidden_units_},
                         dir_path + ".attention.dense.weight." + rank_spec + ".bin",
                         model_file_type);

    if (!use_gptj_residual_) {
        loadWeightFromBin<T>(
            weights_ptr[5], {(size_t)hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    }


    // gate_proj
    loadWeightFromBin<T>(weights_ptr[6],
                         {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.gate_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[7],
                         {(size_t)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.gate_proj.bias." + rank_spec + ".bin",
                         model_file_type);
    
    // up_proj
    loadWeightFromBin<T>(weights_ptr[8],
                         {(size_t)hidden_units_, (size_t)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.up_proj.weight." + rank_spec + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[9],
                         {(size_t)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.up_proj.bias." + rank_spec + ".bin",
                         model_file_type);

    // down_proj
    loadWeightFromBin<T>(weights_ptr[10], {(size_t)(inter_size_ / tensor_para_size_), (size_t)hidden_units_},
                         dir_path + ".mlp.down_proj.weight." + rank_spec + ".bin",
                         model_file_type);



    // 这里的use_gptj_residual_=1 对于gptnoex是默认配置，但是llama的默认设置这里的use_gptj_residual_ = 0
    // 是因为代码是从gptneox复制过来的，就不改动了
    if (use_gptj_residual_) {

        // 这里不走
        loadWeightFromBin<T>(
            weights_ptr[11], {(size_t)hidden_units_}, dir_path + ".mlp.attention.bias.sum.bin", model_file_type);
    }
    else {
        // 走的这个分支
        loadWeightFromBin<T>(
            weights_ptr[11], {(size_t)hidden_units_}, dir_path + ".mlp.down_proj.bias.bin", model_file_type);
    }

    // post_layer_norm
    loadWeightFromBin<T>(
        weights_ptr[12], {(size_t)hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[13], {(size_t)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);

}

template<typename T>
void LlamaDecoderLayerWeight<T>::setWeightPtr()
{

    // std::cout<<"decoder layer weight setWeightPtr"<<std::endl;

    pre_layernorm_weights.beta                            = weights_ptr[0];
    pre_layernorm_weights.gamma                           = weights_ptr[1];
    self_attention_weights.query_weight.kernel            = weights_ptr[2];
    self_attention_weights.query_weight.bias              = weights_ptr[3];
    self_attention_weights.attention_output_weight.kernel = weights_ptr[4];
    self_attention_weights.attention_output_weight.bias   = use_gptj_residual_ ? nullptr : weights_ptr[5];

    // gate_proj
    ffn_weights.intermediate_weight2.kernel = weights_ptr[6];
    ffn_weights.intermediate_weight2.bias   = weights_ptr[7];

    // up_proj
    ffn_weights.intermediate_weight.kernel = weights_ptr[8];
    ffn_weights.intermediate_weight.bias   = weights_ptr[9];

    // down_proj
    ffn_weights.output_weight.kernel       = weights_ptr[10];
    ffn_weights.output_weight.bias         = weights_ptr[11];

    post_attention_layernorm_weights.beta  = weights_ptr[12];
    post_attention_layernorm_weights.gamma = weights_ptr[13];
    is_maintain_buffer                     = true;
}

template<typename T>
void LlamaDecoderLayerWeight<T>::mallocWeights()
{

    // std::cout<<"decoder layer weight malloc weight"<<std::endl;
    // pre_norm
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    // qkv_weight
    deviceMalloc(&weights_ptr[2], hidden_units_ * 3 * hidden_units_ / tensor_para_size_);
    // qkv_weight_bias
    deviceMalloc(&weights_ptr[3], 3 * hidden_units_ / tensor_para_size_);

    // set bias to zero
    deviceMemSetZero(weights_ptr[3], 3 * hidden_units_ / tensor_para_size_); 

    // attention_output_weight weight
    deviceMalloc(&weights_ptr[4], hidden_units_ / tensor_para_size_ * hidden_units_);
    if (!use_gptj_residual_) {
        //attention_output_weight bias
        deviceMalloc(&weights_ptr[5], hidden_units_);
        deviceMemSetZero(weights_ptr[5], hidden_units_); 
    }

    // 暂时还没有支持模型tensor模型并行，所以假定这里的tensor_param_size=1
   
    // gate_proj
    deviceMalloc(&weights_ptr[6], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[7], inter_size_ / tensor_para_size_);
    deviceMemSetZero(weights_ptr[7], inter_size_ / tensor_para_size_); 

    // up_proj
    deviceMalloc(&weights_ptr[8], hidden_units_ * inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[9], inter_size_ / tensor_para_size_);
    deviceMemSetZero(weights_ptr[9], inter_size_ / tensor_para_size_); 

    // down_proj
    deviceMalloc(&weights_ptr[10], inter_size_ / tensor_para_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[11], hidden_units_);
    deviceMemSetZero(weights_ptr[11], hidden_units_); 

    // post_layer_norm
    deviceMalloc(&weights_ptr[12], hidden_units_);
    deviceMalloc(&weights_ptr[13], hidden_units_);
}

template struct LlamaDecoderLayerWeight<float>;
template struct LlamaDecoderLayerWeight<half>;

}  // namespace fastertransformer
