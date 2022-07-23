Pull the repository.
```
git clone https://github.com/ardarslan/score-face.git
```

Create the conda environment and activate it.
```
cd score-face
conda env create -f environment.yml
conda activate score-face
```

Create a folder named "assets" in the main directory of the repository. Download "checkpoint_48.pth" file in it.
```
mkdir assets
cd assets
gdown --id 1-mtdSwuefIZA0n85QWScQo2WRvJNWwUy
```

Run the code.
```
cd ../src
python3 main.py
```