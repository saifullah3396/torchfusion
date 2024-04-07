#!/bin/bash

mkdir -p $DATA_ROOT_DIR/documents/DocBank/DocBank_500K_ori_img
for i in $(seq 1 10); do
    wget "https://layoutlm.blob.core.windows.net/docbank/dataset/DocBank_500K_ori_img.zip.00$i?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D" -O $DATA_ROOT_DIR/documents/DocBank/DocBank_500K_ori_img/part00$i.zip
done
