python merge_beit.py --model beit3_base_patch16_384 --input_size 384 \
--task coco_retrieval --batch_size 16 --sentencepiece_model ./model_weights/beit3.spm \
--data_path YOUR_PATH_HERE --eval --dist_eval --device cuda:0