cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

CONFIG_PATH='config_release/pretrain_SECNet.json'

horovodrun -np 1 python src/pretrain/run_pretrain_SECNet.py \
      --config $CONFIG_PATH \
      --output_dir experiments/SECNet/
