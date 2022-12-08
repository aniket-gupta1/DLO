import torch
from torch.utils.data import DataLoader
from datasets.kitti import kitti
from torch.utils.tensorboard import SummaryWriter
import time
from models.model import DLO_net
from lietorch import SE3, SO3
from config.config import Config
from utils.utils import *
from scipy.spatial.transform import Rotation
import roma

def parse_arguments():
    pass

def loss_fn_so3(pred, gt):
    quat = torch.Tensor(Rotation.from_matrix(gt[:, :3, :3]).as_quat()).to(device)
    R_gt = SO3.InitFromVec(quat)
    R_pred = SO3.exp(pred[:,3:])
    dR = R_gt.inv() * R_pred
    ro_loss = dR.log().norm(dim=-1).sum()

    gt = torch.Tensor(gt).to(device)
    tr_loss = (gt[:,:3,3] - pred[:,:3])**2
    loss = ro_loss + tr_loss.sum()

    return loss

def loss_fn_tf(pred, gt):
    gt = gt.type(torch.float32).cuda()
    # raise ValueError
    quat = roma.rotmat_to_unitquat(gt[:,:3,:3])
    R_gt = SO3.InitFromVec(quat)
    rot_vec_pred = roma.rotmat_to_rotvec(pred[:,:3,:3])
    R_pred = SO3.exp(rot_vec_pred)
    dR = R_gt.inv()*R_pred
    ro_loss = dR.log().norm(dim=-1).sum()

    tr_loss = (gt[:, :3, 3] - pred[:, :3, 3]) ** 2
    loss = ro_loss + tr_loss.sum()

    return loss

def loss_fn(pred, gt, pc):
    pc = pc.squeeze().transpose(1,0).cuda()
    gt = gt.type(torch.float32).cuda()

    pc_tf_pred = torch.add(torch.matmul(pred[:,:3,:3],pc), pred[:,:3,3].unsqueeze(-1).expand(-1, -1, pc.size(-1)))
    pc_tf_gt = torch.add(torch.matmul(gt[:,:3,:3],pc), gt[:,:3,3].unsqueeze(-1).expand(-1, -1, pc.size(-1)))

    # loss = torch.nn.functional.mse_loss(pc_tf_gt, pc_tf_pred)
    loss = torch.mean(torch.abs(pc_tf_gt - pc_tf_pred))

    return loss

def train_epoch(model, optimizer, dataloader,  loss_fn, writer):
    global prev_data
    model.train()

    losses = 0

    for index, data in enumerate(dataloader):
        # print(data['pointcloud'].size())
        # print(data['pose'])
        # print(data['frame_num'])
        # print(data['seq'])
        #
        # time.sleep(200000)
        # print(f"Frame {data['frame_num']} has pointcloud of shape {data['pointcloud'].shape}")

        if data['frame_num']==0:
            prev_data = data
        else:
            optimizer.zero_grad()
            gt = data['pose']
            T = model(prev_data, data)

            loss = loss_fn(T, gt, data['pointcloud'])
            loss.backward(retain_graph=False)
            optimizer.step()

            losses += loss.item()
            writer.add_scalar("Batch Loss/train", loss, index)
            print(f"Batch_index: {index} || Loss: {loss}")

    return losses/(len(dataloader)-1)

def train(cfg, device, writer):
    dataset = kitti(cfg, mode = "training", inbetween_poses = cfg.inbetween_poses,
                    form_transformation = cfg.form_transformation)
    dataloader = DataLoader(dataset, batch_size=1)

    net = DLO_net(cfg, device).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.98), eps=1e-9)

    # loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg.num_epochs):
        tic = time.time()
        loss = train_epoch(net, optimizer, dataloader, loss_fn, writer)
        writer.add_scalar("Loss", loss, epoch)
        print("===========================================================")
        print(f"Epoch: {epoch} || Loss: {loss} || Time: {time.time()-tic}")
        print("===========================================================")

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()

    cfg = Config(512)
    train(cfg, device, writer)
    writer.close()