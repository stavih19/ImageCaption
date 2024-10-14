import torch
from PIL import Image
from get_loader import get_loader, get_transformer
from model import CNNtoRNN

# test our model after training with an image by loading the weights from disk

transform = get_transformer()
train_loader, dataset = get_loader(source_of_train_images)

embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
model.load_state_dict(torch.load('model.pth', map_location=torch.device(device)), strict=False)
model.eval()


test_img1 = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
description = model.caption_image(test_img1.to(device), dataset.vocab)
print(description)
print([description[0], ' '.join(description[1:-1]), description[-1]])