import os
import sys
import glob
import argparse

from generator import DataGenerator
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str)
    parser.add_argument('--ambient', '-an', type=str)
    parser.add_argument('--car', '-cn', type=str)
    args =  parser.parse_args()

    fft_size = 512
    frame_rate = 50
    n_frames = 5

    # initialize data generator
    d_gen = DataGenerator(
            fft_size, 
            frame_rate, 
            n_frames,
            args.source, 
            args.ambient,
            args.car)
