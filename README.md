# RepNet PyTorch
A PyTorch port with pre-trained weights of **RepNet**, from *Counting Out Time: Class Agnostic Video Repetition Counting in the Wild* (CVPR 2020) [[paper]](https://arxiv.org/abs/2006.15418) [[project]](https://sites.google.com/view/repnet) [[notebook]](https://colab.research.google.com/github/google-research/google-research/blob/master/repnet/repnet_colab.ipynb#scrollTo=FUg2vSYhmsT0).

<div align="center">
  <img src="img/example1.gif" height="200" />
  <img src="img/example2.gif" height="200" />
  <img src="img/example3.gif" height="200" />
</div>

## Get Started
- Clone this repo and install dependencies:
```bash
git clone https://github.com/materight/RepNet-pytorch
cd RepNet-pytorch
pip install -r requirements.txt
```

- Download the TensorFlow pre-trained weights and convert them to PyTorch:
```bash
python convert_weights.py
```

## Run inference
Simply run:
```bash
python run.py
```
The script will download a sample video, run inference on it and save the count visualization. You can also specify a video path as argument (either a local path or a YouTube/HTTP URL):
```bash
python run.py --video_path [video_path]
```

Examples of generated videos showing the repetition count and the periodicity score:
