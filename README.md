# Score-Face

### Pull TF_FLAME. Create and activate its conda environment. Install mesh library.

```
git clone git@github.com:ardarslan/TF_FLAME.git
cd TF_FLAME
conda env create -f environment.yml
conda activate tf-flame
git clone git@github.com:MPI-IS/mesh.git
cd mesh
BOOST_INCLUDE_DIRS=/path/to/boost/include make all
cd ..
```

### Prepare data for TF_FLAME:

First, download the texture data.
```
wget http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
unzip FLAME_texture_data.zip
mv texture_data_512.npy data/
mv texture_data_1024.npy data/
mv texture_data_2048.npy data/
rm -rf texture_data_*
rm -rf FLAME_texture_data.zip
```

Second, download the FLAME model.
- Sign in https://flame.is.tue.mpg.de/login.php
- Go to https://flame.is.tue.mpg.de/download.php
- Download "FLAME 2020 (fixed mouth, improved expressions, more data)"
- Create a folder named "models" in the TF_FLAME folder.
- Copy "generic_model.pkl" into "models" folder.

### Fit FLAME on an input image.
Set the following environment variable.
```
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

If you will run the optimization in image space:
```
python3 fit_2D_landmarks.py --texture_data_path ./data/texture_data_512.npy --input_img_path /path/to/input/image
```

If you will run the optimization in texture space:
```
python3 fit_2D_landmarks.py --texture_data_path ./data/texture_data_2048.npy --input_img_path /path/to/input/image
```

### Go into this repository. Create and activate its environment. Prepare its data.
```
cd score-face
conda env create -f environment.yml
conda activate score-face
mkdir assets
cd assets
gdown --id 1sXrlgTC6U2jzWCIUZTbRcY3AbOJfcbbu
gdown --id 1-mtdSwuefIZA0n85QWScQo2WRvJNWwUy
```

### Run the code.

Go into src folder.
```
cd ../src
```

To do optimization in the image space:
```
python3 main.py --input_obj_path /local/home/aarslan/TF_FLAME/results_512/33673.obj --optimization_space image --num_corrector_steps 1 --snr 0.15 --order_views true --two_rounds true
```

To do optimization in the texture space:
```
python3 main.py --input_obj_path /local/home/aarslan/TF_FLAME/results_2048/33673.obj --optimization_space texture --num_corrector_steps 6 --snr 0.015 --order_views true --two_rounds true
```
