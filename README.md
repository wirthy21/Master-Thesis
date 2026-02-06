## Environment Installation
Instruction shows a step by step installation using Anaconda Prompt.

1. Create new conda environment.
```
conda create -n GAIL python=3.10 -y
conda activate GAIL
```

2. Install CUDA enabled pytorch version (Windows + CUDA 12.6).
```
pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
pip install "torchvision==0.21.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

3. Clone repository.
```
git clone https://github.com/wirthy21/Master-Thesis.git
```

4. Install requirements.
```
cd Master-Thesis
pip install -r requirements.txt
```


