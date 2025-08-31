# DomainAdaptationImageClassification

This is the code of my master Thesis.


## Instructions

The following commands are an example execution of the training and evaluation process of the method.

1. Train the Autoencoders

   ```
   python ImageReconstruction.py \
     --train_batch 128\
     --test_batch 32\
     --lr 1e-3 \
     --epochs 30 \
     --dataTrain dataset/MNIST/TRAIN \
     --dataTest dataset/MNIST/TEST \
     --domain Target \
     --device 0
   ```
2. Train Classifiers:

   ```
   python ImageClassification.py \
     --train_batch 128\
     --test_batch 32 \
     --lr 1e-3 \
     --epochs 20 \
     --encoder Results/ImageReconstruction_Source/weights/encoder_best.pth \
     --dataTrain dataset/MNIST/TRAIN \
     --dataTest dataset/MNIST/TEST \
     --domain Source \
     --device 0 \
     --numClasses 10
   ```
3. Train Mapping Network

   ```
   python MlpMapping.py \
     --lr 1e-4\
     --epochs 50\
     --train_batch 16\
     --SourceLatents Results/saveLatents_Source/Latents_Source.pt \
     --TargetLatents Results/saveLatents_Target/Latents_Target.pt \
     --loss ContrastiveLoss \
     --classes 10 \
     --domain S2T \
     --device 0 \
     --centroid True
   ```
4. Domain Adaptation Stage

   ```
   python DomainAdaptation.py \
     --train_batch 256 \
     --test_batch 32 \
     --lr 1e-3 \
     --epochs 30 \
     --encoderSource Results/ImageReconstruction_Source/weights/encoder_best.pth \
     --encoderTarget Results/ImageReconstruction_Target/weights/encoder_best.pth \
     --MLP Results/ImageClassification_Target/weights/classifier_best.pth \
     --classifierTarget Results/ImageClassification_Target/weights/classifier_best.pth \
     --dataTrainTarget dataset/USPS/TRAIN_SUB \
     --dataTrainSource dataset/MNIST/TRAIN \
     --dataTest dataset/USPS/TEST \
     --domain S2T_Triplet \
     --device 0 \
     --numClasses 10
   ```

## Dataset Layout

Source dataset MNIST digits
Target dataset subsampled USPS digits

Images should have the following structure

```
dataset/
    └── MNIST/
        ├── TRAIN/
        │   ├── 0/
        │   ├── 1/
        │   ├── 2/
        │   ├── 3/
        │   ├── 4/
        │   ├── 5/
        │   ├── 6/
        │   ├── 7/
        │   ├── 8/
        │   └── 9/
        └── TEST/
            ├── 0/
            ├── 1/
            ├── 2/
            ├── 3/
            ├── 4/
            ├── 5/
            ├── 6/
            ├── 7/
            ├── 8/
            └── 9/
```

```
dataset/
    └── USPS/
        ├── TRAIN_SUB/
        │   ├── 0/
        │   ├── 1/
        │   ├── 2/
        │   ├── 3/
        │   ├── 4/
        │   ├── 5/
        │   ├── 6/
        │   ├── 7/
        │   ├── 8/
        │   └── 9/
        └── TEST/
            ├── 0/
            ├── 1/
            ├── 2/
            ├── 3/
            ├── 4/
            ├── 5/
            ├── 6/
            ├── 7/
            ├── 8/
            └── 9/
```
