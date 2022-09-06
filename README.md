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

#### Image Space

![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188672385-071435e5-9c49-4a4f-8f72-1b70ac4ba80f.gif)

![optimized_animation](https://user-images.githubusercontent.com/19363284/188672408-fc01b7de-0577-437a-b611-47f3211527a2.gif)


![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188672459-ce2658af-fb2c-4275-ba77-cf5a227d7a4b.gif)

![optimized_animation](https://user-images.githubusercontent.com/19363284/188672474-02c75679-335d-4b94-886f-809c18329aa1.gif)

#### Texture Space

![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188672533-317b7ddf-0a62-48e4-bd75-73270dd8e7b3.gif)

![optimized_animation](https://user-images.githubusercontent.com/19363284/188672548-aa3096cc-2681-4d52-95d5-959d72f60ade.gif)


![unoptimized_animation](https://user-images.githubusercontent.com/19363284/188672576-1167d3dd-8c0a-4478-9f42-58ea60d19915.gif)

![optimized_animation](https://user-images.githubusercontent.com/19363284/188672586-93e11d05-a68f-4324-a36d-04d6d9378692.gif)
