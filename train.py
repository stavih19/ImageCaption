import os.path
import torch
from tqdm import tqdm
import torch.nn as nn
from get_loader import get_loader
from model import CNNtoRNN
import torch.optim as optim


def train(rood_dir_path=source_of_train_images):
    train_loader, dataset = get_loader(rood_dir_path)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_CNN = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 100

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    model.train()

    for epoch in range(num_epochs):
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(rood_dir_path, 'model.pth'))

    return model, dataset
