nohup python3  train_fully_supervised_2D.py  --exp /Fu_LinkNet_46_4_12 --model LinkNetBase --batch_size 64 --gpu 1 --labeled_csv train_046.txt >Fu_LinkNet_46_4_12.log 2>&1 & && \
nohup python3  train_fully_supervised_2D.py  --exp /Fu_UNetdrop_46_4_12 --model UNetWithDrop  --batch_size 16 --gpu 3 --labeled_csv train_046.txt >Fu_UNetWithDrop_46_4_12.log 2>&1 &