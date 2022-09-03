# Score-Face

### Go to the main folder of this repository. Create and activate its conda environment. Prepare its data.
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
python3 main.py --input_obj_path /local/home/aarslan/TF_FLAME/results_512/58232.obj --optimization_space image --order_views true --two_rounds true --num_corrector_steps 1 --snr 0.15
```

To do optimization in the texture space:
```
python3 main.py --input_obj_path /local/home/aarslan/TF_FLAME/results_2048/58232.obj --optimization_space texture --order_views true --two_rounds true --num_corrector_steps 6 --snr 0.015
```
