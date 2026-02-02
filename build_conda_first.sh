source ~/.bashrc
# 1. 创建环境
micromamba create -n slime_pjh_v3 python=3.12 pip -c conda-forge -y
micromamba activate slime_pjh_v3

# 2. 安装 CUDA 编译器和核心库 (这是编译的最底层依赖)
# 脚本中指定了 nvidia/label/cuda-12.9.1
micromamba install -n slime_pjh_v3 cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
micromamba install -n slime_pjh_v3 -c conda-forge cudnn -y

# 3. 设置关键环境变量 (编译时必须能找到 CUDA)
export CUDA_HOME="$CONDA_PREFIX"

# 4. 安装 Python 构建工具 (CMake 和 Ninja 是编译 FlashAttn/Apex 的核心)
# 建议额外加上 packaging, setuptools, wheel 以防万一
# pip install cmake ninja packaging setuptools wheel

# 5. 安装 PyTorch (这是编译扩展的头文件来源)
# 注意：必须在编译 Apex/Flash-Attn 之前安装 PyTorch
pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --extra-index-url https://download.pytorch.org/whl/cu129

export BASE_DIR=/mnt/workspace/users/xintong/pjh # 你的工作目录
mkdir -p $BASE_DIR
cd $BASE_DIR
git clone https://github.com/sgl-project/sglang.git
cd sglang && git checkout 24c91001cf99ba642be791e099d358f4dfe955f5
pip install -e "python[all]"

# ------
