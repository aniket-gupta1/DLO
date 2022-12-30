import torch
from torch.utils.data import DataLoader
from datasets import modelnet
from torch.utils.tensorboard import SummaryWriter
import time
from models.kp_qk_model import DLO_net_single
from config.config import Config
from utils.utils import *
import os
import open3d as o3d

def loss_fn(pred, gt, pc):
    pc = pc.squeeze().transpose(1,0).cuda()
    gt = gt.type(torch.float32).cuda()

    pc_tf_pred = torch.add(torch.matmul(pred[:,:3,:3],pc), pred[:,:3,3].unsqueeze(-1).expand(-1, -1, pc.size(-1)))
    pc_tf_gt = torch.add(torch.matmul(gt[:,:3,:3],pc), gt[:,:3,3].unsqueeze(-1).expand(-1, -1, pc.size(-1)))

    # loss = torch.nn.functional.mse_loss(pc_tf_gt, pc_tf_pred)
    loss = torch.mean(torch.abs(pc_tf_gt - pc_tf_pred)).requires_grad_()

    return loss

def train(cfg, device, writer):
    dataset = modelnet.get_train_datasets(cfg)[0]
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=modelnet.collate_pair)

    model = DLO_net_single(cfg, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # try:

    # loss_fn = torch.nn.MSELoss()
    step=0
    for epoch in range(cfg.num_epochs):
        tic = time.time()

        model.train()

        losses = 0

        for index, data in enumerate(dataloader):
            data = all_to_device(data, device)
            tic2 = time.time()

            optimizer.zero_grad()
            gt = data['pose']
            T = model(data)

            loss = loss_fn(T, gt, data['pointcloud'])
            loss.backward(retain_graph=False)
            optimizer.step()

            losses += loss.item()
            step+=1
            writer.add_scalar("Batch Loss/train", loss, step)
            print(f"Batch_index: {index} || Loss: {loss}")

            print("Time: ", time.time()-tic2)

        scheduler.step()
        loss = losses / (len(dataloader) - 1)
        writer.add_scalar("Loss", loss, epoch)
        print("===========================================================")
        print(f"Epoch: {epoch} || Loss: {loss} || Time: {time.time()-tic}")
        print("===========================================================")

        if epoch % cfg.eval_time == 0:
            path = "../weights"
            if os.path.exists(path):
                model.save(os.path.join(path, f"epoch_{epoch}.pth"))
                model.save(os.path.join(path, f"epoch_latest.pth"))
            else:
                os.makedirs(path)
                # model.eval()
    # except Exception as e:
    #     print(e)
    #     model.save(os.path.join("../weights", f"epoch_latest.pth"))

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    cfg = Config(512)
    train(cfg, device, writer)
    writer.close()
