# Installation

## Requirements:
- conda
- python 2.7
- pytorch 1.0
- cython
- yacs
- scikit-image
- matplotlib (for visualization)
- opencv


## Step-by-step installation:

```
conda create --name afm python=2.7
conda activate afm

pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

cd lib
make
conda develop . ./lib
```