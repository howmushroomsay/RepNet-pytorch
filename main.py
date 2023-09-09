
import os
import cv2
import argparse
import torch
import torchvision.transforms as T

from repnet import utils, plots
from repnet.model import RepNet



parse = argparse.ArgumentParser(description="Run the RepNet model on a given video.")
parse.add_argument("--weights", type=str, default=os.path.join())
parse.add_argument("--video", 
                   type=str, 
                   default="./test.mp4",
                   help='Video to test')
parse.add_argument("--strides", 
                   nargs='+',
                   type=int,
                   default=[1,2,3,4,8],
                   help='Temporal strides to try when testing on the sample video')
parse.add_argument("--device",
                   type=str,
                   default='cuda',
                   help='Device to use for inference')
parse.add_argument("--on-score", 
                   action='store_true',
                   help='If specified, do not plot the periodicity score.')

if __name__ == '__main__':
    args = parse.parse_args()

    if not os.path.exists(args.video):
        print("error path")
        exit(0)

    # Read frames and preprocessing
    print(f'Reading video file and pre_processing frames...')
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((112, 112)),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    raw_frames, frames = [],[]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        raw_frames.append(frame)
        frame = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)
    cap.release()

    # Load model
    model = RepNet()
    state_dict = torch.load(args.weights)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)

    # Test multiple strides and pick the best one
    print('Running inference on multiple stride values...')
    best_stride = None
    best_confidence = None
    best_period_length = None
    best_period_count = None
    best_periodicity_score = None
    best_embeddings = None
    for stride in args.strides:
        # 按步长提取帧
        stride_frames = frames[::stride]
        stride_frames = stride_frames[:(len(stride_frames)//64)*64]
        if len (stride_frames) < 64:
            # 帧数太少不利于判断，直接跳过
            continue
        # 转换成 N x C x D x H x W
        # batch就是64
        stride_frames = torch.stack(stride_frames, axis=0)
        stride_frames = stride_frames.unflatten(0,(-1,64))
        stride_frames = stride_frames.movedim(1,2)
        stride_frames = stride_frames.to(args.device)

        # 进行评估
        raw_period_length = []
        raw_periodicity_score= [] 
        embeddings = []
        with torch.no_grad():
            for i in range(stride_frames.shape[0]):
                