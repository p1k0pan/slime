set -x
source ~/.bashrc
micromamba activate slime_pjh_v3
export BASE_DIR=/mnt/workspace/users/xintong/pjh # 你的工作目录


cd ${BASE_DIR}
git clone https://github.com/fzyzcjy/torch_memory_saver.git
cd torch_memory_saver
git checkout dc6876905830430b5054325fa4211ff302169c6b
pip install . --no-cache-dir --force-reinstall
# pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall

cd ${BASE_DIR}
git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge
git checkout dev_rl
pip install . --no-build-isolation
# pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation

pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

export SLIME_DIR=${BASE_DIR}/slime

cd ${BASE_DIR}
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
cd Megatron-LM/ 
git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
pip install -e . 

cd "$SLIME_DIR"
pip install -e .


pip install "numpy<2"

# apply patch
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/v0.5.7/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.7/megatron.patch

pip install nvidia-cudnn-cu12==9.16.0.29
echo "Conda build completed."