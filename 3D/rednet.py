import torch

class Rednet(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(Rednet, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco1 = Rednet.convblock(self.inChan,  128)
        self.enco2 = Rednet.convblock(128, 128)
        self.enco3 = Rednet.convblock(128, 128)
        self.enco4 = Rednet.convblock(128, 128)
        self.enco5 = Rednet.convblock(128, 128)
        self.enco6 = Rednet.convblock(128, 128)
        self.enco7 = Rednet.convblock(128, 128)
        self.enco8 = Rednet.convblock(128, 128)
        self.enco9 = Rednet.convblock(128, 128)
        self.enco10 = Rednet.convblock(128, 128)
        self.enco11 = Rednet.convblock(128, 128)
        self.enco12 = Rednet.convblock(128, 128)
        self.enco13 = Rednet.convblock(128, 128)
        self.enco14 = Rednet.convblock(128, 128)  

        self.bottleneck = Rednet.convblock(128, 128)

        self.deco14 = Rednet.upconvBlock(128, 128)
        self.deco13 = Rednet.upconvBlock(128, 128)
        self.deco12 = Rednet.upconvBlock(128, 128)
        self.deco11 = Rednet.upconvBlock(128, 128)
        self.deco10 = Rednet.upconvBlock(128, 128)
        self.deco9 = Rednet.upconvBlock(128, 128)
        self.deco8 = Rednet.upconvBlock(128, 128)
        self.deco7 = Rednet.upconvBlock(128, 128)
        self.deco6 = Rednet.upconvBlock(128, 128)
        self.deco5 = Rednet.upconvBlock(128, 128)
        self.deco4 = Rednet.upconvBlock(128, 128)
        self.deco3 = Rednet.upconvBlock(128, 128)
        self.deco2 = Rednet.upconvBlock(128, 128)
        self.deco1 = Rednet.upconvBlock(128, 128)
        self.output = Rednet.upconvBlock(128, self.outChan)
        self.threshold = torch.nn.Threshold(0.5, 0)

    def forward(self, patch):

        enco1 = self.enco1(patch)
        enco2 = self.enco2(enco1)
        enco3 = self.enco3(enco2)
        enco4 = self.enco4(enco3)
        enco5 = self.enco4(enco4)
        enco6 = self.enco4(enco5)
        enco7 = self.enco4(enco6)
        enco8 = self.enco4(enco7)
        enco9 = self.enco4(enco8)
        # enco10 = self.enco4(enco9)
        # enco11 = self.enco4(enco10)
        # enco12 = self.enco4(enco11)
        # enco13 = self.enco4(enco12)
        # enco14 = self.enco4(enco13)
        bottleneck = self.bottleneck(enco9)
        # bottleneck = self.bottleneck(enco14)
        # deco14 = self.deco4(bottleneck) 
        # deco14 += enco14
        # deco13 = self.deco3(deco14) 
        # deco13 += enco13
        # deco12 = self.deco2(deco13) 
        # deco12 += enco12
        # deco11 = self.deco1(deco12) 
        # deco11 += enco11
        # deco10 = self.deco1(deco11) 
        # deco10 += enco10
        deco9 = self.deco1(bottleneck) +  enco9
        deco8 = self.deco1(deco9) + enco8
        deco7 = self.deco1(deco8) + enco7
        deco6 = self.deco1(deco7) + enco6
        deco5 = self.deco1(deco6) + enco5
        deco4 = self.deco1(deco5) + enco4
        deco3 = self.deco1(deco4) + enco3
        deco2 = self.deco1(deco3) + enco2
        deco1 = self.deco1(deco2) + enco1
        output = self.output(deco1) + patch 
        return self.threshold(torch.sigmoid(output))

    def convblock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv3d(inChan, outChan, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose3d(inChan, outChan, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU()) 
    