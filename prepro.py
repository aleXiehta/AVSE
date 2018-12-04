import os
import argparse
import numpy as np

from utils import FacePreprocessor, VideoPreprocessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', '-i', type=str)
    parser.add_argument('--output-path', '-o', type=str)
    parser.add_argument('--predictor-path', '-p', type=str)
    args = parser.parse_args()
    
    # Step 1: Extract frames from videos
    # vp = VideoPreprocessor()
    # vp.extract(args.video_path, args.output_path)

    # Step 2: Crop all mouths from each face
    fp = FacePreprocessor(args.predictor_path)
    fp.gen_cropped_mouth(args.output_path, args.output_path)
    
