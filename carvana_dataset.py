import os, random
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class CarvanaDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            # images
            self.image_folder_path = os.path.join(root_path, "test")
            self.image_path_list =  sorted(os.listdir(self.image_folder_path))
            # masks
            self.image_mask_folder_path = os.path.join(root_path, "test_masks")
            self.image_mask_path_list = sorted(os.listdir(self.image_mask_folder_path))
        
        else:
            # images
            self.image_folder_path = os.path.join(root_path, "train")
            self.image_path_list =  sorted(os.listdir(self.image_folder_path))
            # masks
            self.image_mask_folder_path = os.path.join(root_path, "train_masks")
            self.image_mask_path_list = sorted(os.listdir(self.image_mask_folder_path))
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_folder_path, self.image_path_list[index])).convert("RGB")
        mask = Image.open(os.path.join(self.image_mask_folder_path, self.image_mask_path_list[index])).convert("L")

        return self.transform(img), self.transform(mask)
        
    def __len__(self):
        return len(self.image_mask_path_list)
    
if __name__ == "__main__":

    # Test the function in the dataset.

    root_path = "D:\CarvanaDataset_tiny"

    car_dataset = CarvanaDataset(root_path, False)

    print(f"There are {car_dataset.__len__()} images in the dataset.")
    random_index = random.randint(0, car_dataset.__len__())

    img_tensor, mask_tensor = car_dataset.__getitem__(random_index)
    tensor_to_pil = transforms.ToPILImage()
    img = tensor_to_pil(img_tensor)
    mask = tensor_to_pil(mask_tensor).convert("RGB")

    output_image = Image.new('RGB', (img.size[0] + mask.size[0], max(img.size[1], mask.size[1])))
    output_image.paste(img, (0,0))
    output_image.paste(mask, (img.size[0], 0))
    output_image.show()  