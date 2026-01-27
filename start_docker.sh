chmod +x run_geo3k_vlm.sh
# chmod -R u+rwX /ltstorage/home/pan/slime/outputs
mkdir -p /ltstorage/home/pan/slime/outputs
chmod 777 /ltstorage/home/pan/slime/outputs


docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -u root \
  -v /ltstorage/home/pan/data2/model:/root/model:ro \
  -v /ltstorage/home/pan/slime/outputs:/root/slime/outputs \
  -v ./run_geo3k_vlm.sh:/root/slime/run_geo3k_vlm.sh \
  -e SLIME_SCRIPT_MODEL_NAME=Qwen3-VL-4B-Thinking \
  -it slimerl/slime:latest \
  /bin/bash -c "cd /root/slime && ./run_geo3k_vlm.sh"
