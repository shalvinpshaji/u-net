import torch
import torchvision.transforms
import dice_loss
import model
import dataset
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm


epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = 'tumor_data/kaggle_3m/'
images = glob.glob(path+'*/*[0-9].tif')
mask = []
for image in images:
    mask.append(image.split('.')[0]+"_mask.tif")


U_Net = model.UNet(in_filter=3, down_filters=[64, 128, 256, 512], up_filters=[512, 256, 128, 64]).to(device)

transforms = torchvision.transforms.ToTensor()
tumor_dataset = dataset.TumorDataset(images=images, masks=mask, transforms=transforms)
loader = DataLoader(tumor_dataset, 2, shuffle=True)

optimizer = torch.optim.Adam(U_Net.parameters(), lr=3e-4)

d_loss = dice_loss.DiceLoss()
for i in range(1, epochs+1):
    for sample in tqdm(loader):
        image = sample[0].to(device)
        mask = sample[1].to(device)
        optimizer.zero_grad()
        prediction = U_Net(image)
        loss = d_loss(prediction, mask)
        loss.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {i} : {loss}")
