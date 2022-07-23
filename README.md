Pull the repository.
```
git clone https://github.com/ardarslan/score-face.git
```

Download assets.zip, extract it to assets folder, put it in the main folder.
```

```

Create conda environments.
```
conda env create -f flame_environment.yml
conda env create -f score_face_environment.yml
```

Activate flame environment. Install mesh library. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable. Run TF_FLAME on an example image.
```
conda env activate flame
cd src/flame/mesh
BOOST_INCLUDE_DIRS=/path/to/boost/include make all
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ..
python3 fit_2D_landmarks.py --source_img_path ../../assets/40044.png
```

Go to nbs folder. Open main.ipynb. Choose score-face as the kernel. Run the notebook.