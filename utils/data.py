from torch.utils.data import Dataset
import torch
import os
from PIL import Image

class CustomImageDataset(Dataset):

    def __init__(self, rootDir, transform=None) -> None:
        self.rootDir = rootDir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        self.folders = os.listdir(self.rootDir)
        for label in self.folders:
            class_folder = os.path.join(self.rootDir,str(label))
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.img_paths.append(img_path)
                self.labels.append(int(label))
        
    def __len__(self) -> int:
        return len(self.img_paths) 

    def __getitem__(self, idx) -> any:

        sample_img_path = self.img_paths[idx]
        sample_label = self.labels[idx]
        sample_img = Image.open(sample_img_path).convert("L")
        if self.transform:
            sample_img = self.transform(sample_img)

        # Convert label to a tensor
        sample_label = torch.tensor(sample_label, dtype=torch.long)
        return sample_img, sample_label



# if __name__ == "__main__":

#     datasetDir = "D:\Pantelis-Files\MSc Thesis\Domain Adaptation\code\Thesis_example_code_v1\dataset\MNIST\TRAIN"
#     dataset = CustomImageDataset(datasetDir)
#     labels = dataset.labels
#     import numpy
#     L = numpy.unique(labels)
#     print(L)
# #     for i, sample in enumerate(dataset):
#         pdb.set_trace()
#         if i == 3:
#             break
#     # a = dataset.__getitem__('4')
#     # cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
#     # cv2.imshow("Output",a)
#     # cv2.waitKey()
#     pdb.set_trace()
#     dataloader = DataLoader(dataset,4,True)
#     pdb.set_trace()
#     for batch_idx, (images,labels) in enumerate(dataloader):
#         pdb.set_trace()

