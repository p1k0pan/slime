set -x
# -----------------------------
# 1. 安装编译型依赖 (FlashAttn, TE, Apex)
# -----------------------------
source ~/.bashrc
micromamba activate slime_pjh_v3
pip install cmake ninja  # 再次确认
export BASE_DIR=/mnt/workspace/users/xintong/pjh # 你的工作目录

MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
# pip install git+https://github.com/ISEEKYAN/mbridge.git --no-deps

pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.0


# Apex (最容易挂的，必须带参数)
# NVCC_APPEND_FLAGS="--threads 4" \
#   pip -v install --disable-pip-version-check --no-cache-dir \
#   --no-build-isolation \
#   --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
#   git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

cd ${BASE_DIR} 
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .