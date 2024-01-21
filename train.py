import torch
from torch import optim, nn 
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm 

from unet import UNet
from carvana_dataset import CarvanaDataset

if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 2
    DATA_PATH = ""
    MODEL_SAVE_PATH = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CarvanaDataset(DATA_PATH)

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

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
            





