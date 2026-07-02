import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 320


class FFTTransform:
    def __call__(self, img):

        img = np.array(img)  # H,W,3

        fft_channels = []

        for c in range(3):

            channel = img[:, :, c]

            fft = np.fft.fft2(channel)
            fft_shift = np.fft.fftshift(fft)

            magnitude = np.log(np.abs(fft_shift) + 1)

            magnitude = (
                (magnitude - magnitude.min()) /
                (magnitude.max() - magnitude.min() + 1e-8)
            )

            magnitude = (magnitude * 255).astype(np.uint8)

            fft_channels.append(magnitude)

        fft_img = np.stack(fft_channels, axis=2)

        return Image.fromarray(fft_img)
    




class TwoStreamResNet(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.rgb_net = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        self.fft_net = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )

        feat_dim = self.rgb_net.fc.in_features

        self.rgb_net.fc = nn.Identity()
        self.fft_net.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(feat_dim * 2, num_classes)
        )

    def forward(self, rgb, fft):

        rgb_feat = self.rgb_net(rgb)
        fft_feat = self.fft_net(fft)

        fused = torch.cat([rgb_feat, fft_feat], dim=1)

        return self.classifier(fused)
    


# -----------------------
# 1. Handle input argument
# -----------------------
if len(sys.argv) != 2:
    print("Usage: python test.py <image_path>")
    exit()

image_path = sys.argv[1]

# -----------------------
# 2. Recreate the SAME model
# -----------------------

model = TwoStreamResNet(num_classes=2)



# Load weights
model.load_state_dict(torch.load("mona_resnet50.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

class_names = ["Authentic", "Forged"]

# -----------------------
# 3. Define transforms
# -----------------------
rgb_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

fft_transform = transforms.Compose([
    FFTTransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# -----------------------
# 4. Load image
# -----------------------
try:
    img = Image.open(image_path).convert("RGB")
except:
    print("Error: Could not open image:", image_path)
    exit()

rgb_img = rgb_transform(img).unsqueeze(0).to(DEVICE)
fft_img = fft_transform(img).unsqueeze(0).to(DEVICE)


# -----------------------
# 5. Predict
# -----------------------
with torch.no_grad():
    output = model(rgb_img, fft_img)

    probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
    pred = output.argmax(1).item()

print(f"Predicted class: {class_names[pred]}")
print(f"Authentic probability: {probs[0]:.4f}")
print(f"Forged probability: {probs[1]:.4f}")