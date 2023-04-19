import torch

class Ournetconv(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(Ournetconv, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco3 = Ournetconv.block(self.inChan,   64)
        self.enco2 = Ournetconv.block(64,  128)
        self.enco1 = Ournetconv.block(128, 256)

        self.conv3 = Ournetconv.convBlock(64,  64)
        self.conv2 = Ournetconv.convBlock(128, 128)
        self.conv1 = Ournetconv.convBlock(256, 256)

        self.bottleneck = Ournetconv.block(256, 512)

        self.upconv1 = Ournetconv.upconvBlock(512, 256)
        self.upconv2 = Ournetconv.upconvBlock(256, 128)
        self.upconv3 = Ournetconv.upconvBlock(128, 64)

        self.deco1 = Ournetconv.block(512, 256)
        self.deco2 = Ournetconv.block(256, 128)
        self.deco3 = Ournetconv.block(128, 64)

        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv3d(64, self.outChan, kernel_size=1)

    def forward(self, patch):
        enco3 = self.enco3(patch)
        enco2 = self.enco2(self.conv3(enco3))
        enco1 = self.enco1(self.conv2(enco2))
        bottleneck = self.bottleneck(self.conv1(enco1))
        tmp = self.upconv1(bottleneck)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))
        return self.threshold(torch.sigmoid(self.output(deco3)))

    def block(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm3d(outChan),
        torch.nn.PReLU(),
        torch.nn.Conv3d(outChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm3d(outChan),
        torch.nn.PReLU())

    def convBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=2, stride=2))  

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose3d(inChan, outChan, kernel_size=2, stride=2))  
