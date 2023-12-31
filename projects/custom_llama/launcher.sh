lightning run model ./train_pl.py \
    --strategy=fsdp \
    --devices=8 \
    --accelerator=cuda \
    --precision="32"