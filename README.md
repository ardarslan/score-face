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
python3 main.py --input_obj_path INPUT_OBJ_PATH --optimization_space image --order_views true --two_rounds true --num_corrector_steps 1 --snr 0.15
```

To do optimization in the texture space:
```
python3 main.py --input_obj_path INPUT_OBJ_PATH --optimization_space texture --order_views true --two_rounds true --num_corrector_steps 6 --snr 0.015
```

### Results

Optimization in the Image Space

![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188681577-df15b06a-e0b5-4d80-a577-13117dabd78e.gif) ![optimized_animation](https://user-images.githubusercontent.com/19363284/188681601-b0928a26-8608-46be-b041-a5d98a1bcc39.gif)


![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188681632-567cacd5-3b2f-4585-a922-66951db6a00d.gif) ![optimized_animation](https://user-images.githubusercontent.com/19363284/188681653-37b81f91-4f25-4353-ac85-191bd1674d50.gif)


![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188964020-b18d32e6-c6af-44c3-a213-7fa3fa2aa1f9.gif) ![optimized_animation](https://user-images.githubusercontent.com/19363284/188963990-7608c7d3-4063-4d28-9ce2-62010df5429f.gif)

![ezgif-3-3d001e314f](https://user-images.githubusercontent.com/19363284/189201636-8591b98a-c808-4064-a0a8-facbdfd7a901.gif) ![ezgif-3-bbc31152a3](https://user-images.githubusercontent.com/19363284/189201656-e83933ef-8e84-4055-bd0c-ff84abf0aaf2.gif)


Optimization in the Texture Space

![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188672533-317b7ddf-0a62-48e4-bd75-73270dd8e7b3.gif) ![optimized_animation](https://user-images.githubusercontent.com/19363284/188672548-aa3096cc-2681-4d52-95d5-959d72f60ade.gif)


![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188672576-1167d3dd-8c0a-4478-9f42-58ea60d19915.gif) ![optimized_animation](https://user-images.githubusercontent.com/19363284/188672586-93e11d05-a68f-4324-a36d-04d6d9378692.gif)


![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188965520-ddb0beb9-fec0-48ab-9334-4536c4461fac.gif) ![optimized_animation](https://user-images.githubusercontent.com/19363284/188965538-97d3209c-a40d-448b-bc94-c70887eb59c5.gif)


![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188964154-cb8a2ab1-aa7b-458b-af2c-140999e9e854.gif) ![optimized_animation](https://user-images.githubusercontent.com/19363284/188964176-1ab2c2b2-67f2-441f-9f28-457d91320e0c.gif)

Note: The code under the directory https://github.com/ardarslan/score-face/tree/master/src/score_sde was adapted from https://github.com/yang-song/score_sde_pytorch.
