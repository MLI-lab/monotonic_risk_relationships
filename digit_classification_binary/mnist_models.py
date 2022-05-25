import re

import torch
from torch import nn

from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201

num_classes = 10
kernel_size = 5
num_in_channels = 1


create_model_function_lut = {
    "alexnet": alexnet,
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "resnext50_32x4d": resnext50_32x4d,
    "resnext101_32x8d": resnext101_32x8d,
    "wide_resnet50_2": wide_resnet50_2,
    "wide_resnet101_2": wide_resnet101_2,
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
}


class NetMNIST(nn.Module):
    """
    """
    def __init__(self, model_id, num_classes=num_classes):
        super().__init__()
        create_model_function = create_model_function_lut[model_id]
        self.dim_output = num_classes
        self.model = create_model_function(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def output_to_target(self, output):
        _, predicted_target = torch.max(output, 1)
        return predicted_target


class AlexNetMNIST(NetMNIST):
    """
    """
    def __init__(self, model_id, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model.features[0] = torch.nn.Conv2d(num_in_channels, 64, kernel_size=kernel_size, stride=4, padding=2)



class VGGMNIST(NetMNIST):
    """
    """
    def __init__(self, model_id, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model.features[0] = torch.nn.Conv2d(num_in_channels, 64, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=(1, 1), bias=False)


class ResNetMNIST(NetMNIST):
    """
    """
    def __init__(self, model_id, **kwargs):
        super().__init__(model_id, **kwargs)
        self.model.conv1 = torch.nn.Conv2d(num_in_channels, 64, kernel_size=(kernel_size, kernel_size), stride=(2, 2), padding=(3, 3), bias=False)


class DenseNetMNIST(NetMNIST):
    """
    """
    def __init__(self, model_id, **kwargs):
        super().__init__(model_id, **kwargs)
        num_init_features = 64
        if model_id in ["densenet161"]:
            num_init_features = 96
        self.model.features[0] = torch.nn.Conv2d(num_in_channels, num_init_features, kernel_size=(kernel_size, kernel_size), stride=(2, 2), padding=(3, 3), bias=False)


class MLPMNIST(nn.Module):
    """
    Multi-layered perceptron (MLP)
    """
    def __init__(self, model_id, num_classes=num_classes):
        super().__init__()
        self.dim_output = num_classes
        self.dropout_rate = 0.5
        self.parse_config(model_id)

        # Define hidden layers
        layers = []
        dim_input = 75 * 75
        for n, dropout in zip(self.dim_hidden_layers, self.dropouts):
            layers.append(nn.Linear(dim_input, n))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(p=self.dropout_rate))
            dim_input = n
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.dim_hidden_layers[-1], self.dim_output)

    def parse_config(self, model_id):
        config = model_id.split("_")
        self.dim_hidden_layers = []
        self.dropouts = []
        for cfg in config[1:]:
            dim = int(re.findall(r"\d+", cfg)[0])
            dropout = re.findall(r"[a-z]+", cfg)
            if len(dropout) == 0:
                dropout = False
            elif dropout[0] == 'd':
                dropout = True 
            self.dim_hidden_layers.append(dim)
            self.dropouts.append(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def output_to_target(self, output):
        _, predicted_target = torch.max(output, 1)
        return predicted_target


class RFNMNIST(nn.Module):
    """
    Random convolution feature network (RFN)
    """
    def __init__(self, model_id, num_classes=num_classes):
        super().__init__()
        self.dim_output = num_classes
        self.parse_config(model_id)

        # Define conv layers
        input_height = 75
        input_width = 75
        in_channels = 1
        if self.conv_kernel_size is None:
            self.conv_kernel_size = 6
        if self.pool_kernel_size is None:
            self.pool_kernel_size = 16
        feature_height = input_height // self.pool_kernel_size
        feature_width = input_width // self.pool_kernel_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, self.num_filters, kernel_size=self.conv_kernel_size, padding="same"),
            torch.nn.AvgPool2d(self.pool_kernel_size, stride=self.pool_kernel_size)
        )
        self.classifier = nn.Linear(feature_height*feature_width*self.num_filters, self.dim_output)

        # Initialize convolutional layers and freeze weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                m.weight.requires_grad = False

    def parse_config(self, model_id):
        config = model_id.split('_')
        cfg = re.findall(r"\d+[a-z]*", config[0])[0]
        self.num_filters = int(re.findall(r"\d+", cfg)[0])
        unit = re.findall(r"[a-z]+", cfg)
        if len(unit) == 1 and unit[0] == 'k':
            self.num_filters *= 1000
        self.conv_kernel_size = None
        self.pool_kernel_size = None
        if len(config) > 1:
            self.conv_kernel_size = int(config[1])
        if len(config) > 2:
            self.pool_kernel_size = int(config[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def output_to_target(self, output):
        _, predicted_target = torch.max(output, 1)
        return predicted_target


class RidgeClassifierMNIST(nn.Module):
    """
    Ridge classifier
    """
    def __init__(self, model_id, num_classes=num_classes):
        super().__init__()
        self.dim_output = 1 # num_classes is not used, model output is a scalar
        self.parse_config(model_id)

        # Define layers
        dim_input = 75 * 75
        self.classifier = nn.Linear(dim_input, self.dim_output)

    def parse_config(self, model_id):
        self.regularization_param = int(re.findall(r"\d+", model_id)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def regularization(self):
        # L2 penalty on weight only, not on bias
        return self.regularization_param * (self.classifier.weight**2).sum()

    def output_to_target(self, output):
        # squeeze the last dimension because output has input shape (batch_size, 1)
        predicted_target = (torch.sign(output.squeeze()) + 1) / 2.0
        return predicted_target


class LassoClassifierMNIST(nn.Module):
    """
    Lasso classifier
    """
    def __init__(self, model_id, num_classes=num_classes):
        super().__init__()
        self.dim_output = 1 # num_classes is not used, model output is a scalar
        self.parse_config(model_id)

        # Define layers
        dim_input = 75 * 75
        self.classifier = nn.Linear(dim_input, self.dim_output)

    def parse_config(self, model_id):
        self.regularization_param = int(re.findall(r"\d+", model_id)[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def regularization(self):
        # L1 penalty on weight only, not on bias
        return self.regularization_param * self.classifier.weight.abs().sum()

    def output_to_target(self, output):
        # squeeze the last dimension because output has input shape (batch_size, 1)
        predicted_target = (torch.sign(output.squeeze()) + 1) / 2.0
        return predicted_target


class BinaryClassificationL2Loss:
    """
    L2 loss assuming {-1, 1} encoding of {0, 1}-valued targets
    """
    def __init__(self):
        pass

    def __call__(self, input, target):
        # Compute L2 loss after converting target from {0, 1} to {-1, 1}
        l2_loss = torch.mean((input - (2*target - 1))**2)
        return l2_loss


class LogisticRegressionMNIST(nn.Module):
    """
    Logistic regression
    """
    def __init__(self, model_id, num_classes=num_classes, threshold=0.5):
        super().__init__()
        self.dim_output = 1 # num_classes is not used, model output is a scalar
        self.threshold = threshold
        self.parse_config(model_id)

        # Define layers
        dim_input = 75 * 75
        self.classifier = nn.Linear(dim_input, self.dim_output)

    def parse_config(self, model_id):
        """
        Args:
            model_id: Examples: logisticregression_l1_lambda, logisticregression_l2_lambda
        """
        config = model_id.split("_")
        if len(config) > 1:
            self.penalty = config[1]
            self.regularization_param = int(config[2])
        else:
            self.penalty = None
            self.regularization_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.classifier(x).squeeze()
        x = torch.sigmoid(x)
        return x

    def regularization(self):
        if self.penalty is None or self.regularization_param is None:
            regularization = torch.tensor(0.0)
        else:
            # L1 or L2 penalty on weight only, not on bias
            if self.penalty == "l1":
                regularization = self.regularization_param * self.classifier.weight.abs().sum()
            elif self.penalty == "l2":
                regularization = self.regularization_param * (self.classifier.weight**2).sum()
            else:
                raise ValueError("Penalty {} in model_id unknown.".format(self.penalty))
        return regularization

    def output_to_target(self, output):
        # squeeze the last dimension because output has input shape (batch_size, 1)
        predicted_target = (torch.sign(output.squeeze() - self.threshold) + 1) / 2.0
        return predicted_target


