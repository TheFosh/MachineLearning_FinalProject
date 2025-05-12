import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.io import read_image
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, is_train, transform=None, target_transform=None):
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
        self.images = torch.empty(len(self.annotations), 3, image_size, image_size)
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

        self.in_to_h1 = nn.Conv2d(3, 32, (5, 5), padding=(2, 2))  # 32 x 224 x 224
        self.h1_to_h2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))  # 32 x 224 x 224
        # Dropout 10%
        # Maxpool2d -> 32 x 112 x 112
        self.h2_to_h3 = nn.Conv2d(32, 8, (3, 3), padding=(1, 1))  # 8 x 112 x 112
        # Maxpool2d -> 8 x 56 x 56
        self.h3_to_h4 = nn.Linear(8 * 56 * 56, 20)  # 20
        self.h4_to_out = nn.Linear(20, 53)  # 53

    def forward(self, x):
        if torch.cuda.is_available():
            device = next(self.parameters()).device  # gets model's device (CPU or CUDA)
            x = x.to(device)

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
    print(f"Cuda Available? {torch.cuda.is_available()}")
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

#bar graph of top predictions, where the indices of values corresponds to those of labels
def plot_predictions(labels, values, title="Predictions"):
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Card")
    plt.ylabel("Confidence")
    plt.show()

def predict_card(image_path, trained_model_path="cardNN.pt"):
    card_mapping = {
        0: "ace of clubs",
        1: "ace of diamonds",
        2: "ace of hearts",
        3: "ace of spades",
        4: "eight of clubs",
        5: "eight of diamonds",
        6: "eight of hearts",
        7: "eight of spades",
        8: "five of clubs",
        9: "five of diamonds",
        10: "five of hearts",
        11: "five of spades",
        12: "four of clubs",
        13: "four of diamonds",
        14: "four of hearts",
        15: "four of spades",
        16: "jack of clubs",
        17: "jack of diamonds",
        18: "jack of hearts",
        19: "jack of spades",
        20: "joker",
        21: "king of clubs",
        22: "king of diamonds",
        23: "king of hearts",
        24: "king of spades",
        25: "nine of clubs",
        26: "nine of diamonds",
        27: "nine of hearts",
        28: "nine of spades",
        29: "queen of clubs",
        30: "queen of diamonds",
        31: "queen of hearts",
        32: "queen of spades",
        33: "seven of clubs",
        34: "seven of diamonds",
        35: "seven of hearts",
        36: "seven of spades",
        37: "six of clubs",
        38: "six of diamonds",
        39: "six of hearts",
        40: "six of spades",
        41: "ten of clubs",
        42: "ten of diamonds",
        43: "ten of hearts",
        44: "ten of spades",
        45: "three of clubs",
        46: "three of diamonds",
        47: "three of hearts",
        48: "three of spades",
        49: "two of clubs",
        50: "two of diamonds",
        51: "two of hearts",
        52: "two of spades"
    }

    # resize image
    transform = transforms.Compose([transforms.Resize((224, 224))])
    image_tensor = transform(read_image(image_path) / 255.0).unsqueeze(0)

    # Load the model
    model = CardNetwork()
    model.load_state_dict(torch.load(trained_model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        sorted_output, indices = torch.sort(model(image_tensor), descending=True)
        print(f"Output: {output}")
        print(f"Indices: {indices}")
        top_five = [card_mapping.get(x.item()) for x in indices[0][:5]]
        print(f"Top 5 Predictions: {top_five}")
        plot_predictions(top_five, sorted_output[0][0:5])
        prediction = card_mapping.get(torch.argmax(output, dim=1).item())

    print(f"Prediction: {prediction}")

#trainNN(epochs=25, batch_size=128, display_test_acc=True)
predict_card("ImageData\\valid\\ten of diamonds\\2.jpg")
#predict_card("ImageData\\custom\\002.jpg")