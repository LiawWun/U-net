import torch
from torch import optim, nn 
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import os 

from unet import UNet
from carvana_dataset import CarvanaDataset

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script for UNet on Carvana dataset")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data_folder', type=str, default="D:\\CarvanaDataset_tiny", help='Path to the Carvana dataset')
    parser.add_argument('--model_save_folder', type=str, default="D:\\CarvanaDataset_tiny\\model", help='Path to save the trained model')

    return parser.parse_args()

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == "__main__":

    args = parse_arguments()

    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DATA_FOLDER = args.data_folder
    MODEL_SAVE_FOLDER = args.model_save_folder

    create_folder_if_not_exists(MODEL_SAVE_FOLDER) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CarvanaDataset(DATA_FOLDER)

    generator = torch.Generator().manual_seed(2024)
    train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2], generator)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):

        

        # Training section
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            pred = model(img)
            optimizer.zero_grad()

            loss = criterion(pred, mask)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        valid_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(valid_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                pred = model(img)
                loss = criterion(pred, mask)

                valid_running_loss += loss.item()
            valid_loss = valid_running_loss / (idx + 1)
            
        print("=" * 50)
        print(f"Epoch: {epoch + 1}")
        print(f"Train : {train_loss:.4f} Valid loss: {valid_loss:.4f}")
        print("=" * 50)
        save_name = "unet_" + str(epoch) + ".pth"
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_FOLDER, save_name))
            





