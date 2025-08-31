from utils.utils import createDirectory
import random
import argparse
import os
import pdb
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--inputPath", type=str, default="dataset/USPS/train", help="Path to dataset to subsample from")
parser.add_argument("--outputPath", type=str, default="dataset/USPS/train_sub", help="Path to dataset to subsample from")
parser.add_argument("--rate", type=float, default=0.25, help="Sample rate")
opt = parser.parse_args()

if __name__ == "__main__":
    InputPath = os.path.join(os.getcwd(), opt.inputPath)
    OutputPath = os.path.join(os.getcwd(), opt.outputPath)
    createDirectory(OutputPath)
    
    rate = opt.rate
    classes = os.listdir(InputPath)
    

    for idx,cls in enumerate(classes):
        copied = 0
        tmpInPath = os.path.join(InputPath,cls)
        samples = os.listdir(tmpInPath) # contains initial images of class cls
        targetSize = round(len(samples)*rate)

        tmpOutpath = os.path.join(OutputPath,cls)
        createDirectory(tmpOutpath)
        print(f"Copying {targetSize} samples out of {len(samples)} of class {cls}")
        while (copied < targetSize):
            randomIdx = random.randint(0, len(samples)-1) # randomly choose sample to copy to target susbsample path
            # print(f"randomIdx = {randomIdx} len(samples) = {len(samples)}")
            source = os.path.join(InputPath, cls, samples[randomIdx])
            target = os.path.join(OutputPath, cls, samples[randomIdx])
            shutil.copyfile(source, target)
            samples.pop(randomIdx) # remove selected sample
            copied += 1





