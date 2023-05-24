

ckpt_path=/home/lll/models/ft_gptj_6b/1-gpu/
lib_path=/home/lll/tmp_code/FasterTransformer/build/lib/libth_transformer.so
vocab_file=/home/lll/tmp_code/FasterTransformer/models/gpt6B/vocab.json
merges_file=/home/lll/tmp_code/FasterTransformer/models/gpt6B/merges.txt

# python3 gpt_example.py --layer_num 28 --output_len 1 --size_per_head 256 \
#             --vocab_size 50400 --inference_data_type fp16 --weights_data_type fp16 \
#             --lib_path $lib_path \
#             --vocab_file $vocab_file \
#             --merges_file $merges_file \
#             --ckpt_path $ckpt_path


python3 multi_gpu_gpt_example.py --layer_num 28 --output_len 1 --size_per_head 256 \
            --vocab_size 50400 --inference_data_type fp16 --weights_data_type fp16 \
            --lib_path $lib_path \
            --vocab_file $vocab_file \
            --merges_file $merges_file \
            --ckpt_path $ckpt_path --use_gpt_decoder_ops
