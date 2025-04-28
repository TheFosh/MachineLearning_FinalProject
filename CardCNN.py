import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.io import read_image

class MNIST(Dataset):
    def __init__(self):
        # https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
        df = pd.read_csv("card_train.csv")
        self.train_numbers = torch.tensor(df.iloc[:, 1:].to_numpy() / 255.0, dtype=torch.float).view(-1, 1, 224, 224)
        self.train_labels = torch.tensor(df["label"].to_numpy())

        df = pd.read_csv("card_test.csv")
        self.test_numbers = torch.tensor(df.iloc[:, 1:].to_numpy() / 255.0, dtype=torch.float).view(-1, 1, 224, 224)
        self.test_labels = torch.tensor(df["label"].to_numpy())

        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_numbers[item], self.train_labels[item]

    def __len__(self):
        return self.len


class NumberClassify(nn.Module):
    def __init__(self):
        # Call the constructor of the super class
        super(NumberClassify, self).__init__()

        self.in_to_h1 = nn.Conv2d(1, 32, (5, 5), padding=(2, 2))  # 32 x 224 x 224
        self.h1_to_h2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))  # 32 x 224 x 224
        # Maxpool2d -> 32 x 112 x 112
        self.h2_to_h3 = nn.Conv2d(32, 8, (3, 3), padding=(1, 1))  # 8 x 112 x 112
        # Maxpool2d -> 8 x 56 x 56
        self.h3_to_h4 = nn.Linear(8 * 56 * 56, 20)  # 20
        self.h4_to_out = nn.Linear(20, 52)  # 52

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
    mnist = MNIST()

    # create data loader
    mnist_loader = DataLoader(mnist, batch_size=batch_size, drop_last=True, shuffle=True)

    # determine which device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create CNN
    number_classify = NumberClassify().to(device)
    if trained_network is not None:
        number_classify.load_state_dict(trained_network)
        number_classify.train()

    print(f"Total parameters: {sum(param.numel() for param in number_classify.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(number_classify.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(mnist_loader)):
            x, y = data

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = number_classify(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        if display_test_acc:
            with torch.no_grad():
                predictions = torch.argmax(number_classify(mnist.test_numbers.to(device)), dim=1)  # Get the prediction
                correct = (predictions == mnist.test_labels.to(device)).sum().item()
                print(f"Accuracy on test set: {correct / len(mnist.test_labels):.4f}")

    torch.save(number_classify.state_dict(), save_file)

def predict_card(card_jpg=None, trained_model_path="cardNN.pt"):
    if card_jpg is None:
        print("No image.")
        return

    model = NumberClassify()
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    #use CNN to predict card here

trainNN(epochs=5, display_test_acc=True)
#predict_card()