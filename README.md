# Score-Face

Face Texture Reconstruction and Synthesis with Score-Based Generative Models

## Pull repositories

Pull score-face, TF_FLAME, and mesh repositories (mesh should be inside TF_FLAME directory).

```
cd
git clone https://github.com/ardarslan/score-face.git
git clone https://github.com/TimoBolkart/TF_FLAME.git
cd TF_FLAME
git clone https://github.com/MPI-IS/mesh.git
```

## Setup mesh and TF_FLAME repositories

Deactivate current environments.
```
conda deactivate
conda deactivate
deactivate
```

Switch to new software stack. And load required modules.
```
env2lmod
module load gcc/6.3.0 boost/1.74.0 eth_proxy python_gpu/3.7.4
```

Go into TF_FLAME directory, and create a virtual environment, and activate it.
```
mkdir .virtualenvs
python3 -m venv .virtualenvs/TF_FLAME
source .virtualenvs/TF_FLAME/bin/activate
```

Go into mesh directory, and install mesh.
```
pip install -U pip
BOOST_INCLUDE_DIRS=/cluster/apps/gcc-6.3.0/boost-1.74.0-yl65iuwmyxsiyxehki4zjnued4nubqyn/include make all
```

Go into TF_FLAME directory. Change chumpy version from 0.69 to 0.70. Delete ipython. Then install requirements.
```
cd ..
----- Change chumpy version from 0.69 to 0.70. And delete ipython. -----
pip install -r requirements.txt
```

Sign in https://flame.is.tue.mpg.de/login.php

Go to https://flame.is.tue.mpg.de/download.php

Download FLAME 2020 (fixed mouth, improved expressions, more data)

Copy generic_model.pkl into TF_FLAME/models.

Download http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip

Copy texture_data_256.npy into TF_FLAME/data.

## Setup score-face repository

Deactivate current environments.
```
conda deactivate
conda deactivate
deactivate
```

Go into score-face directory.
```
cd
cd score-face
conda env create -f environment.yml
conda activate score_face
mkdir -p exp/ve/ffhq_256_ncsnpp_continuous
gdown 1-mtdSwuefIZA0n85QWScQo2WRvJNWwUy -O exp/ve/ffhq_256_ncsnpp_continuous/checkpoint_48.pth
```

## Prepare data

Deactivate current environments.
```
conda deactivate
conda deactivate
deactivate
```

Activate score_face conda environment.
```
conda activate score_face
```

Untar the dataset.
```
mkdir -p /cluster/scratch/$(whoami)/FFHQ
tar -C /cluster/scratch/$(whoami)/FFHQ -xvf /cluster/project/infk/hilliges/buehlmar/datasets/FFHQ/raw.tar
```

Resize images.
```
cd scripts
python3 resize_ffhq_images.py --input_images_dir /cluster/scratch/$(whoami)/FFHQ/raw --output_images_dir /cluster/scratch/$(whoami)/FFHQ/resized --output_images_height 256 --output_images_width 256
```

Deactivate current environment.
```
conda deactivate
```

Execute the following lines. NOTE: Python file should be changed so that it extracts mesh and texture files for all images (not only a single one).
```
module load gcc/6.3.0 boost/1.74.0 eth_proxy python_gpu/3.7.4
cd
cd TF_FLAME
source .virtualenvs/TF_FLAME/bin/activate
bsub -n 4 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python fit_2D_landmarks.py --model_fname './models/generic_model.pkl' --flame_lmk_path './data/flame_static_embedding.pkl' --texture_mapping './data/texture_data_256.npy' --target_img_path '/cluster/scratch/aarslan/FFHQ/resized/00009.png' --out_path '/cluster/scratch/aarslan/FFHQ/mesh_and_texture' --visualize False
```

## Render a textured mesh / Play with SDE notebook:

Deactivate current environments.
```
conda deactivate
conda deactivate
deactivate
```

Use nbs/render_textured_mesh.ipynb
