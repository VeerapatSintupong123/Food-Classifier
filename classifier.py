import base64
from io import BytesIO
from PIL import Image
import torch
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet34_Weights

class FoodClassifier:
    def __init__(self, model_path, class_file, foods, device=None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.foods = foods
        self.to_class = self._load_class_mapping(class_file)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model = self._build_model(len(foods))
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_class_mapping(self, class_file):
        to_class = {}
        cnt = 0
        with open(class_file, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                food = line.strip()
                if food in self.foods:
                    to_class[cnt] = food
                    cnt += 1
        return to_class

    def _build_model(self, n_class):
        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Sequential(
            self._get_fc_layers(fc_sizes=[(512, 300), (300, 100)], ps=[0.5, 0.5]),
            nn.Linear(100, n_class)
        )
        return model

    @staticmethod
    def _get_fc_layers(fc_sizes, ps):
        layers = []
        for fc_size, p in zip(fc_sizes, ps):
            layers.extend([
                nn.Linear(fc_size[0], fc_size[1]),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(fc_size[1]),
                nn.Dropout(p=p)
            ])
        return nn.Sequential(*layers)

    def decode_base64_to_image(self, base64_str: str) -> Image:
        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))
            return img
        except Exception as e:
            raise ValueError("Invalid base64 string")

    def classify(self, base64_image: str):
        # Decode the base64 string to image
        image = self.decode_base64_to_image(base64_image)
        
        # Apply transformations
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            label = torch.argmax(output, dim=1).cpu().numpy()[0]

        return self.to_class[label]
