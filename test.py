from randlanet import RandLANet
import torch

if __name__=="__main__":
    pc = torch.randn((1,200,3))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RandLANet(None, device)

    encoder(pc)
