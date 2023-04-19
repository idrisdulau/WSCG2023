import torch

class Ournetmaxp(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(Ournetmaxp, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco3 = Ournetmaxp.block(self.inChan,   64)
        self.enco2 = Ournetmaxp.block(64,  128)
        self.enco1 = Ournetmaxp.block(128, 256)

        self.bottleneck = Ournetmaxp.block(256, 512)

        self.upconv1 = Ournetmaxp.upconvBlock(512, 256)
        self.upconv2 = Ournetmaxp.upconvBlock(256, 128)
        self.upconv3 = Ournetmaxp.upconvBlock(128, 64)

        self.deco1 = Ournetmaxp.block(512, 256)
        self.deco2 = Ournetmaxp.block(256, 128)
        self.deco3 = Ournetmaxp.block(128, 64)

        self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv3d(64, self.outChan, kernel_size=1)

    def forward(self, patch):
        enco3 = self.enco3(patch)
        enco2 = self.enco2(self.pool(enco3))
        enco1 = self.enco1(self.pool(enco2))
        bottleneck = self.bottleneck(self.pool(enco1))
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

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose3d(inChan, outChan, kernel_size=2, stride=2))  
