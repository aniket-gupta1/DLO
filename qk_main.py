import torch
from torch.utils.data import DataLoader
from datasets.kitti import kitti
from torch.utils.tensorboard import SummaryWriter
import time
from models.qk_model import DLO_net_single
from lietorch import SE3, SO3
from config.config import Config
from utils.utils import *
from scipy.spatial.transform import Rotation
#import roma
import os
import open3d as o3d


def check_correctness(prev, curr):
    print(curr['pose'])
    T = curr['pose'].cpu().numpy().reshape(4,4).astype(np.float32)
    # T = curr['pose'].cpu().numpy().squeeze().astype(np.float32)
    pc1 = prev['pointcloud'].cpu().numpy().squeeze().astype(np.float32)
    pc2 = curr['pointcloud'].cpu().numpy().squeeze().astype(np.float32)

    print(type(pc1))
    print(pc1.shape)
    print(pc2.shape)
    print(T[:3,3])
    # pts =  T[:3,:3] @ pc2.T + np.expand_dims(T[:3, 3], -1)
    pts =  T[:3,:3] @ pc2.T + np.array([[T[2,3]], [-T[0,3]], [T[1,3]]])
    print(type(pts))
    print(pts.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.T)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)

    o3d.visualization.draw_geometries([pcd])

    return False


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
    loss = torch.mean(torch.abs(pc_tf_gt - pc_tf_pred)).requires_grad_()

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
            T = model(data)
        else:
            optimizer.zero_grad()
            gt = data['pose']
            T = model(data)

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

    model = DLO_net_single(cfg, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # try:

    # loss_fn = torch.nn.MSELoss()
    step=0
    for epoch in range(cfg.num_epochs):
        tic = time.time()
        # loss = train_epoch(net, optimizer, dataloader, loss_fn, writer)

        model.train()

        losses = 0

        for index, data in enumerate(dataloader):
            tic2 = time.time()


            if data['frame_num'] == 0:
                # prev_data = data
                T = model(data)
            else:
                # if not check_correctness(data, data):
                #     raise ValueError
                optimizer.zero_grad()
                gt = data['pose']
                # T = model(prev_data, data)
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
