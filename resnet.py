import torch
import torch.nn as nn

#Module: clase base para todos los modulosde redes neuronales
class BasicBlock1d(nn.Module):
    expansion = 1
                        #implanes=64
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        #Durante el entrenamiento, pone a cero aleatoriamente algunos de los 
        # elementos del tensor de entrada con probabilidad 
        # p utilizando muestras de una distribución de Bernoulli.
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    #parte externa de la arquitectura
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        #Aplica una convulacion 1D sobre una señal de entrada compuesra por varios planes de entrada. 
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        #aceleracion de entrenamiento de redes profundas.
        self.bn1 = nn.BatchNorm1d(inplanes)
        #aplica la funcion de unidad lineal rectificada por elementos
        self.relu = nn.ReLU(inplace=True)
        #aplica una agrupacion 1D max.
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #Se ejecutan cada una de la capas. El valor de salida de cada capa es el valor de entrada de la siguiente.
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        #Aplica un agrupamiento promedio adaptativo 1D sobre una señal 
        # de entrada compuesta por varios planos de entrada.
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        #Aplica una transformación lineal a los datos entrantes
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #Aplica un agrupamiento promedio adaptativo 1D 
        # sobre una señal de entrada compuesta por varios
        #  planos de entrada.
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        #Concatena la secuencia dada de seqtensores en la dimensión dada. Todos los tensores deben tener la misma forma
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model
