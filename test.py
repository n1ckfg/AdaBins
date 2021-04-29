import os
import argparse

from infer import InferenceHelper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', 
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model', 
        default='nyu',
        help='model type, kitti or nyu'
    )  
        
    args = parser.parse_args()

    infer_helper = InferenceHelper(dataset=args.model)

    # predict depths of images stored in a directory and store the predictions in 16-bit format in a given separate dir
    infer_helper.predict_dir(args.input, args.output)
