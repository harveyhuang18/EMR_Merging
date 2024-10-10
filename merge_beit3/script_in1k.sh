python merge_beit.py --model beit3_base_patch16_224 --task imagenet --batch_size \
64 --sentencepiece_model ./model_weights/beit3.spm --data_path \
YOUR_PATH_HERE \
--output_dir ./output --eval --dist_eval --device cuda:3