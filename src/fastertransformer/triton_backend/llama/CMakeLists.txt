cmake_minimum_required(VERSION 3.8)

set(parallel_gpt_triton_backend_files
    LlamaTritonModel.cc
    LlamaTritonModelInstance.cc
)

add_library(LlamaTritonBackend STATIC ${parallel_gpt_triton_backend_files})
set_property(TARGET LlamaTritonBackend PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(LlamaTritonBackend PRIVATE TransformerTritonBackend Llama tensor memory_utils -lcublasLt)
target_compile_features(LlamaTritonBackend PRIVATE cxx_std_14)