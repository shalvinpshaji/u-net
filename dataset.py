import torchvision.transforms.transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision


class TumorDataset(Dataset):

    def __init__(self, images: list[str], masks: list[str], transforms):
        super(TumorDataset, self).__init__()
        self.data_x = images
        self.data_y = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data_x)

    def __getitem__(self, item: int) -> dict:
        image = Image.open(self.data_x[item])
        mask = Image.open(self.data_y[item])
        image = self.transforms(image)
        mask = self.transforms(mask)
        image = torchvision.transforms.Resize((572, 572))(image)
        mask = torchvision.transforms.Resize((388, 388))(mask)
        sample = [image, mask]
        return sample
