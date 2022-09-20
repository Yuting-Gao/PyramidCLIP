pip install ftfy regex dareblopy pandas google protobuf pyyaml scipy

cp /youtu_pedestrian_detection/datasets/imagenet/val.tar /dev/shm/
tar -xf /dev/shm/val.tar -C /dev/shm
rm /dev/shm/val.tar

python3 -um torch.distributed.launch --nnodes=$HOST_NUM --nproc_per_node=$HOST_GPU_NUM \
--node_rank=$INDEX --master_port=3111 --master_addr=$CHIEF_IP \
main.py \
--visual_model RN50 \
--batch_size_test 256 \
--test_dataset imagenet \
--test_data_path /dev/shm/ \
--precision fp32 \
--evaluate /youtu_pedestrian_detection/yutinggao/models/pyramidclipforrelease/PyramidCLIP-YFCC15MV2-RN50.pth

