python merge_beit.py --model beit3_base_patch16_480 --input_size 480 \
--task coco_captioning --batch_size 32 --sentencepiece_model \
./model_weights/beit3.spm --data_path YOUR_PATH_HERE \
--output_dir ./output --eval --dist_eval --device cuda:2