python merge_beit.py --model beit3_base_patch16_224  --task nlvr2 --batch_size 32 --sentencepiece_model \
./model_weights/beit3.spm --data_path YOUR_PATH_HERE --eval --dist_eval --device cuda:3