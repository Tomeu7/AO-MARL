import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


class DenoisingAutoencoderLinear(nn.Module):
    def __init__(self,
                 num_inputs=28 * 28,
                 hidden_dim1=500,
                 hidden_dim2=120,
                 hidden_dim3=40):
        super(DenoisingAutoencoderLinear, self).__init__()

        # Enconder
        self.linear_encoder1 = nn.Linear(num_inputs, hidden_dim1)
        self.linear_encoder2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear_encoder3 = nn.Linear(hidden_dim2, hidden_dim3)

        # Decoder
        self.linear_decoder1 = nn.Linear(hidden_dim3, hidden_dim2)
        self.linear_decoder2 = nn.Linear(hidden_dim2, hidden_dim1)
        self.linear_decoder3 = nn.Linear(hidden_dim1, num_inputs)

        self.ReLU = nn.ReLU()

    def encoder(self, x):
        x = self.ReLU(self.linear_encoder1(x))
        x = self.ReLU(self.linear_encoder2(x))
        x = self.ReLU(self.linear_encoder3(x))
        return x

    def decoder(self, x):
        x = self.ReLU(self.linear_decoder1(x))
        x = self.ReLU(self.linear_decoder2(x))
        x = self.ReLU(self.linear_decoder3(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DenoisingAutoencoderCnnCentroids(nn.Module):
    def __init__(self,
                 num_inputs=(28, 28),
                 hidden_dim1=500,
                 hidden_dim2=120,
                 hidden_dim3=40):
        # conv3d
        # (N, CIN, DIN, HIN, WIN)
        # (N, COUT, DOUT, HOUT, WOUT)
        # DOUT=[DIN+2PADDING-DILATION*(KERNEL_SIZE-1)-1]/STRIDE + 1

        super(DenoisingAutoencoderCnnCentroids, self).__init__()
        # input (128, 1, 16, 16, 64)
        self.encoder1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)  # (128, 16, 16, 16, 64)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)  # (128, 16, 8, 8, 32)
        self.encoder2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)  # (128, 32, 8, 8, 32)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)  # (128, 32, 4, 4, 16)
        self.encoder3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)  # (128, 64, 4, 4, 16)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)  # (128, 32, 2, 2, 8) 64*2*2*8

        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, 128)

    def cnn(self, x):
        x = F.relu(self.encoder1(x))
        x = self.maxpool1(x)
        x = F.relu(self.encoder2(x))
        x = self.maxpool2(x)
        x = F.relu(self.encoder3(x))
        x = self.maxpool3(x)
        return x

    def feedforward(self, x):
        x = F.relu(self.linear1(x.view(-1, 2048)))
        x = self.linear2(x)
        return x

    def forward(self, x):
        x = self.cnn(x)
        x = self.feedforward(x)
        return x

class DenoisingAutoencoderCNN(nn.Module):
    def __init__(self, num_inputs=(28, 28), hidden_dim1=500, hidden_dim2=120, hidden_dim3=40):
        # conv3d
        # (N, CIN, DIN, HIN, WIN)
        # (N, COUT, DOUT, HOUT, WOUT)
        # DOUT=[DIN+2PADDING-DILATION*(KERNEL_SIZE-1)-1]/STRIDE + 1

        super(DenoisingAutoencoderCNN, self).__init__()

        self.encoder1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        self.encoder2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.encoder3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.decoder1 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1)
        #print("mmm")

    def encoder(self, x):
        x = F.relu(self.encoder1(x))
        x = self.maxpool1(x)
        x = F.relu(self.encoder2(x))
        x = self.maxpool2(x)
        x = self.encoder3(x)
        return x

    def decoder(self, x):
        x = F.relu(self.decoder1(x))
        x = F.relu(self.decoder2(x))
        x = self.decoder3(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DenoisingAutoencoderCNN2DSingleSubapeture(nn.Module):
    def __init__(self,
                 criterion='MSE',
                 batch_norm=False):

        super(DenoisingAutoencoderCNN2DSingleSubapeture, self).__init__()
        # input (128, 1, 16, 16)
        self.encoder1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # (128, 128, 16, 16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # (128, 16, 8, 8, 32)
        self.encoder2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # (128, 256, 8, 8, 32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # (128, 32, 4, 4, 16)
        self.encoder3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # (128, 512, 4, 4, 16)
        self.decoder1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.criterion = criterion
        self.batch_norm = batch_norm

        if batch_norm:
            self.encoder_bn1 = nn.BatchNorm2d(128)
            self.encoder_bn2 = nn.BatchNorm2d(256)
            self.encoder_bn3 = nn.BatchNorm2d(512)
            self.decoder_bn1 = nn.BatchNorm2d(256)
            self.decoder_bn2 = nn.BatchNorm2d(128)
        else:
            self.encoder_bn1 = None
            self.encoder_bn2 = None
            self.encoder_bn3 = None
            self.decoder_bn1 = None
            self.decoder_bn2 = None

    def encoder(self, x):

        if self.batch_norm:
            x = F.relu(self.encoder1(x))
            x = self.encoder_bn1(self.maxpool1(x))
            x = F.relu(self.encoder2(x))
            x = self.encoder_bn2(self.maxpool2(x))
            x = self.encoder_bn3(F.relu(self.encoder3(x)))
        else:
            x = F.relu(self.encoder1(x))
            x = self.maxpool1(x)
            x = F.relu(self.encoder2(x))
            x = self.maxpool2(x)
            x = F.relu(self.encoder3(x))

        return x

    def decoder(self, x):
        # print(x.shape)
        if self.batch_norm:
            x = self.decoder_bn1(F.relu(self.decoder1(x)))
            x = self.decoder_bn2(F.relu(self.decoder2(x)))
        else:
            # print(x.shape)
            x = F.relu(self.decoder1(x))
            # print(x.shape)
            x = F.relu(self.decoder2(x))
        x = self.decoder3(x)
        # print(x.shape)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.criterion == "BCE":
            x = torch.sigmoid(x)
        return x

class Autoencoder:
    def __init__(self, config):
        self.type = config.autoencoder['type'].lower()

        if self.type == "cnn":
            self.model = DenoisingAutoencoderCNN(num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40) # num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40)
        elif self.type == "cnn_single_subaperture":
            self.model = DenoisingAutoencoderCNN2DSingleSubapeture()  # num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40)
        elif self.type == "cnn_centroids":
            self.model = DenoisingAutoencoderCnnCentroids()
        else:
            self.model = DenoisingAutoencoderLinear(num_inputs=16*16*64, hidden_dim1=500, hidden_dim2=120, hidden_dim3=40)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.autoencoder['path'] is not None:
            autoencoder_path = config.autoencoder['path']
            if torch.cuda.is_available():
                 self.model.load_state_dict(torch.load(autoencoder_path))
            else:
                self.model.load_state_dict(torch.load(autoencoder_path, map_location=torch.device('cpu')))
            self.model.to(device=self.device)
            self.model.eval()

    def predict(self, noisy_tensor, only_inference_time=False):  # noise_image
        # noisy_tensor = torch.tensor(noisy_image).to(device=self.device)
        if self.type == "cnn":
            with torch.no_grad():
                predicted = self.model(noisy_tensor.view(1,1,16,16,-1)).cpu().numpy().reshape(16,16,-1)
        elif self.type == "cnn_single_subaperture":
            with torch.no_grad():
                if only_inference_time:
                    predicted = self.model(noisy_tensor)
                else:
                    predicted = self.model(noisy_tensor.view(-1,1,16,16)).cpu().numpy().reshape(-1,16,16)
        elif self.type == "cnn_centroids":
            with torch.no_grad():
                predicted = self.model(noisy_tensor.view(1,1,16,16,-1)).cpu().numpy()
        else:
            with torch.no_grad():
                # print(noisy_tensor.shape)
                predicted = self.model(noisy_tensor.reshape(-1,16*16*64)).cpu().numpy().reshape(16,16,-1)
        return predicted

# autoencoder = autoencoder({"type":"linear", "path":"autoencoder"})
# image2predict = np.load("noise3_image_scao_sh_10x10_16pix_2m_gs9_noise3_delay0.npy")
# real_image2predict = image2predict[0,:,:]
# tensor_image = torch.Tensor(real_image2predict)
# with torch.no_grad():
#    predicted = autoencoder.predict(tensor_image.view(-1,160*160)).cpu().numpy().reshape(160,160)
# plt.imshow(predicted)
# plt.show()