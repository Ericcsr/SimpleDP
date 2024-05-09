# SimpleDP
A simplified version of diffusion policy

## Install
```
pip install -r requirements.txt
```

## Train
```
python train.py
```

## Test
```
python test.py
```

## Info
1. Training and testing specifications are defined in `push_t_image.json` and `push_t_state.json`
2. To collect data use `collect_data.py --o <outputname>.zarr`