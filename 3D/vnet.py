import torch

class Vnet(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(Vnet, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco4 = Vnet.singleBlock(self.inChan,   16)
        self.enco3 = Vnet.singleBlock(32,  32)
        self.enco2 = Vnet.doubleBlock(64,  64)
        self.enco1 = Vnet.doubleBlock(128,  128)

        self.conv4 = Vnet.convBlock(16,  32)
        self.conv3 = Vnet.convBlock(32,  64)
        self.conv2 = Vnet.convBlock(64,  128)
        self.conv1 = Vnet.convBlock(128, 256)

        self.bottleneck = Vnet.doubleBlock(256, 256)

        self.upconv1 = Vnet.upconvBlock(256, 128)
        self.upconv2 = Vnet.upconvBlock(128, 64)
        self.upconv3 = Vnet.upconvBlock(64,  32)
        self.upconv4 = Vnet.upconvBlock(32,  16)    

        self.deco1 = Vnet.doubleBlock(256, 128)
        self.deco2 = Vnet.doubleBlock(128, 64)
        self.deco3 = Vnet.singleBlock(64,  32)
        self.deco4 = Vnet.outputBlock(32,  self.outChan)

        self.threshold = torch.nn.Threshold(0.5, 0)

    def forward(self, patch):
        enco4 = self.enco4(patch)
        enco3 = self.enco3(self.conv4(enco4))
        enco2 = self.enco2(self.conv3(enco3))
        enco1 = self.enco1(self.conv2(enco2))
        bottleneck = self.bottleneck(self.conv1(enco1)) 
        tmp = self.upconv1(bottleneck)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))
        tmp = self.upconv4(deco3)
        deco4 = self.deco4(torch.cat((tmp, enco4), dim=1))
        return self.threshold(torch.sigmoid(deco4))

    def singleBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=5, stride=1, padding=2),
        torch.nn.BatchNorm3d(outChan),
        torch.nn.PReLU())
    
    def doubleBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=5, stride=1, padding=2),
        torch.nn.BatchNorm3d(outChan),
        torch.nn.PReLU(),
        torch.nn.Conv3d(outChan, outChan, kernel_size=5, stride=1, padding=2),
        torch.nn.BatchNorm3d(outChan),
        torch.nn.PReLU())

    def outputBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=1))

    def convBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=2, stride=2))  

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose3d(inChan, outChan, kernel_size=2, stride=2))   
