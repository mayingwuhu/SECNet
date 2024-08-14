cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH


CONFIG_PATH='config_release/msrvtt_ret.json'

TXT_DB='data/msrvtt_ret/txt/test.jsonl'
IMG_DB='data/msrvtt_ret/videos'

horovodrun -np 1 python src/tasks/run_video_retrieval.py \
      --do_inference 1 \
      --inference_split test \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 64 \
      --output_dir  output/ \
      --config $CONFIG_PATH
