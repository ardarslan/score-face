# Score-Face

Pull DECA. Create and activate its conda environment. Prepare its data.
```
git clone https://github.com/ardarslan/DECA.git
cd DECA
bash install_conda.sh
conda activate deca-env
bash fetch_data.sh
```

Fit 3DMM on the input image.
```
python3 demos/demo_reconstruct.py -i PATH_TO_INPUT_IMAGE_FOLDER --saveObj True
```

Go into this repository. Create and activate its environment. Prepare its data.
```
cd score-face
conda env create -f environment.yml
conda activate score-face
mkdir assets
cd assets
gdown --id 1-mtdSwuefIZA0n85QWScQo2WRvJNWwUy
```

Run the code.
```
cd ../src
python3 main.py
```