from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds


def eval(model, split, seq_length, n_cpu, disp, device):
    dataset = GolfDB(data_file='/Users/davidromero/Documents/Capstone/Elaboration F24/ML/golfdb-master/train_split_{}.pkl'.format(split),
                     vid_dir='/Users/davidromero/Documents/Capstone/Elaboration F24/ML/golfdb-master/data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'].to(device), sample['labels'].to(device)  # Send to device (GPU or CPU)
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch)  # Model will run on the appropriate device
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()  # Move data back to CPU for processing
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze().cpu())  # Ensure labels are on CPU for processing
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split = 1
    seq_length = 64
    n_cpu = 6

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('/Users/davidromero/Documents/Capstone/Elaboration F24/ML/golfdb-master/models/swingnet_1800.pth.tar', map_location=device)  # Load model to the correct device
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)  # Send model to GPU or CPU
    model.eval()

    PCE = eval(model, split, seq_length, n_cpu, True, device)  # Pass device to eval function
    print('Average PCE: {}'.format(PCE))