import os
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, is_train, transform=transforms.Grayscale(), target_transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        #set labels to dataframe containing only correct rows
        if is_train:
            self.annotations = self.annotations[self.annotations['data set'] == 'train']
        else:
            self.annotations = self.annotations[self.annotations['data set'] == 'test']

        images = self.annotations['filepaths'].to_numpy()
        image_size = 224
        self.images = torch.empty(len(self.annotations), 1, image_size, image_size)
        for i in range(len(self.annotations)):
            image = read_image("ImageData\\" + images[i])
            if self.transform:
                image = self.transform(image)
            self.images[i] = image

        an = self.annotations.copy()
        self.img_labels = torch.tensor(an["class index"].to_numpy())


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 1])
        #print(str(img_path))
        image = read_image(str(img_path))/255
        #print(f"Success: {str(img_path)}")
        label = self.annotations.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CardNetwork(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(CardNetwork, self).__init__()

        self.in_to_h1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))  # 32 x 224 x 224
        self.h1_to_h2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))  # 32 x 224 x 224
        # Maxpool2d -> 32 x 112 x 112
        self.h2_to_h3 = nn.Conv2d(32, 8, (3, 3), padding=(1, 1))  # 8 x 112 x 112
        # Maxpool2d -> 8 x 56 x 56
        self.h3_to_h4 = nn.Linear(8 * 56 * 56, 20)  # 20
        self.h4_to_out = nn.Linear(20, 53)  # 52

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))  # 32 x 224 x 224
        x = F.relu(self.h1_to_h2(x))  # 32 x 224 x 224
        x = F.dropout2d(x, 0.1)  # drop out 10% of the channels
        x = F.max_pool2d(x, (2, 2))  # 32 x 112 x 112
        x = F.relu(self.h2_to_h3(x))  # 8 x 112 x 112
        x = F.max_pool2d(x, (2, 2))  # 8 x 56 x 56
        x = torch.flatten(x, 1)  # flatten all dimensions except batch -> 8 * 56 * 56
        x = F.relu(self.h3_to_h4(x))  # 20
        return self.h4_to_out(x)  # 52

def trainNN(epochs=5, batch_size=16, lr=0.001, display_test_acc=False, trained_network=None, save_file="cardNN.pt"):

    # load dataset
    train_cards = CustomImageDataset("ImageData\\cards.csv", "ImageData", is_train=True)
    test_cards = CustomImageDataset("ImageData\\cards.csv", "ImageData", is_train=False)

    # create data loader
    train_loader = DataLoader(train_cards, batch_size=batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_cards, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    card_classify = CardNetwork().to(device)
    if trained_network is not None:
        card_classify.load_state_dict(trained_network)
        card_classify.train()
    print(f"Total parameters: {sum(param.numel() for param in card_classify.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(card_classify.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(train_loader)):
            x, y = data

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = card_classify(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            with torch.no_grad():
                predictions = torch.argmax(card_classify(test_cards.images), dim=1)  # Get the prediction
                correct = (predictions == test_cards.img_labels.to(device)).sum().item()
                print(f"Accuracy on test set: {correct / len(test_cards.img_labels):.4f}")
    torch.save(card_classify.state_dict(), save_file)

trainNN(epochs=1, batch_size= 2, display_test_acc=True)