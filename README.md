# patternRecognition

## Start

```
git submodule update --init --recursive
```

conda或者python管理环境

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # GPU
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # CPU

cd sam2
pip install -e .
pip install opencv-python matplotlib

cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

输入和输出文件夹

会有3个输出，score最高的会输出到best_output文件夹，每个输出都会输出到output文件夹(其实不需要，后面的效果很差)

```
patterRecognition
├── raw
├── best_output
├── output
├── sam2
└── inference.py
```

运行推理

```
export PYTHONPATH="sam2" # 或者最好设置为绝对路径
python inference.py # 会运行所有在raw中的
```