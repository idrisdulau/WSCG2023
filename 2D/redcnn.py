import torch

class Redcnn(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(Redcnn, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco1 = Redcnn.convblock(self.inChan,  96)
        self.enco2 = Redcnn.convblock(96, 96)
        self.enco3 = Redcnn.convblock(96, 96)
        self.enco4 = Redcnn.convblock(96, 96)

        self.bottleneck = Redcnn.convblock(96, 96)

        self.deco4 = Redcnn.upconvBlock(96, 96)
        self.deco3 = Redcnn.upconvBlock(96, 96)
        self.deco2 = Redcnn.upconvBlock(96, 96)
        self.deco1 = Redcnn.upconvBlock(96, 96)

        self.output = Redcnn.upconvBlock(96, self.outChan)

        self.threshold = torch.nn.Threshold(0.5, 0)

    def forward(self, patch):
        enco1 = self.enco1(patch)
        enco2 = self.enco2(enco1)
        enco3 = self.enco3(enco2)
        enco4 = self.enco4(enco3)
        bottleneck = self.bottleneck(enco4)
        deco4 = self.deco4(bottleneck) + enco4
        deco3 = self.deco3(deco4)
        deco2 = self.deco2(deco3) +  enco2
        deco1 = self.deco1(deco2)
        output = self.output(deco1) + patch 
        return self.threshold(torch.sigmoid(output))

    def convblock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv2d(inChan, outChan, kernel_size=5, stride=1, padding=0),
        torch.nn.ReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inChan, outChan, kernel_size=5, stride=1, padding=0),
        torch.nn.ReLU())
