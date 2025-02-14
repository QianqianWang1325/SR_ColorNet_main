import argparse
import os
import random
import sys
import time

import cv2
import numpy as np
import pytorch_msssim as torchssim
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from liuling_sr_color_work1.liuling_sr_color import fjn_util, iqa
# from liuling_sr_color.work1.Net import net_optimizer
# from liuling_sr_color.work1.write import excel_write
import model
import fjn_util 
import iqa
import net_optimizer
from write import excel_write

torch.cuda.set_device(0)
device = torch.cuda.current_device()
print(f"当前代码在第{device}张显卡上运行。")


def set_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_train_pan_path', type=str, default='../../WQQ/TCASR/data/train/pan2/')
    parser.add_argument('--img_train_ms_path', type=str, default='../../WQQ/TCASR/data/train/ms2/')
    parser.add_argument('--img_val_pan_path', type=str, default='../../WQQ/TCASR/data/test/pan2/')
    parser.add_argument('--img_val_ms_path', type=str, default='../../WQQ/TCASR/data/test/ms2/')
    parser.add_argument('--img_test_pan_path', type=str, default='../../WQQ/TCASR/data/test/pan2/')
    parser.add_argument('--img_test_ms_path', type=str, default='../../WQQ/TCASR/data/test/ms2/')

    parser.add_argument('--train_pkl_path', type=str, default='./log_1/pkl/train/')
    parser.add_argument('--best_pkl_path', type=str, default='./log_1/pkl/best/')
    parser.add_argument('--process_path', type=str, default='./log_1/process/')

    parser.add_argument('--test_result_path', type=str, default='./log_1/supple/test_result/')
    parser.add_argument('--test_out_result_path', type=str, default='./log_1/supple/out/')
    parser.add_argument('--tensorboard_path', type=str, default='./log_1/tensorboard/')
    parser.add_argument('--test_statistics_path', type=str, default='./log_1/statistics/test/')

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--all_epoch', type=int, default=32)
    parser.add_argument('--val_step', type=int, default=4)

    parser.add_argument('--normalization', type=str2bool, default=True)
    parser.add_argument('--data_range', type=int, default=1)

    return parser.parse_args()


def str2bool(v):
    return v.lower() in ('true')


def train(parameters):
    # tensorboard可视化：文件保存路径
    tensorboard_writer = SummaryWriter(parameters.tensorboard_path)

    # 1,加载训练数据集
    train_dl = Data_Loader(pan_src=parameters.img_train_pan_path,
                           ms_src=parameters.img_train_ms_path, batch_size=parameters.batch_size, shuffle=True,
                           rand_rotate=True, rand_flip=True,
                           normalzero2one=parameters.normalization).load()
    val_dl = Data_Loader(pan_src=parameters.img_test_pan_path,
                         ms_src=parameters.img_test_ms_path, batch_size=1, shuffle=True, rand_rotate=True,
                         rand_flip=True,
                         normalzero2one=parameters.normalization).load()

    # 2, 加载模型
    train_model = model.Net(parameters.device).to(parameters.device)
    train_model.load_pkl(parameters.train_pkl_path, mode='trained', load=True)

    l1_loss_fun = nn.L1Loss().to(parameters.device)
    ssim_fun = torchssim.SSIM(data_range=parameters.data_range, channel=3)

    train_loss_count = fjn_util.AverageCounter()
    train_ssim_count = fjn_util.AverageCounter()

    val_loss_count = fjn_util.AverageCounter()
    val_ssim_count = fjn_util.AverageCounter()
    val_psnr_count = fjn_util.AverageCounter()
    val_mae_count = fjn_util.AverageCounter()
    val_mse_count = fjn_util.AverageCounter()
    val_pearsonr_count = fjn_util.AverageCounter()

    optimizer = net_optimizer.Adam_Oprimizer(train_model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    for epoch in range(train_model.train_epoch, parameters.all_epoch + 1):
        train_loss_count.reset()
        train_ssim_count.reset()
        train_model.train()
        with tqdm(total=len(train_dl), ascii=True) as t:
            t.set_description(('epoch: {}/{}'.format(epoch, parameters.all_epoch)))
            for step, (label_ab, label_l, input_l, ms_img_o) in enumerate(train_dl):
                input_l = input_l.float().to(parameters.device)
                label_l = label_l.float().to(parameters.device)
                label_ab = label_ab.float().to(parameters.device)
                label_lab = torch.cat([label_l, label_ab], dim=1)

                # 3.3 计算损失，反向传播
                out_l, out_ab = train_model(input_l)
                out_lab = torch.cat((out_l, out_ab), dim=1)
                l1_loss = l1_loss_fun(out_l, label_l) + l1_loss_fun(out_ab, label_ab)

                loss = l1_loss - ssim_fun(out_lab, label_lab)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for i in range(label_l.shape[0]):
                    train_ssim_count.update(
                        ssim_fun(out_lab[i, :, :, :].unsqueeze(0), label_lab[i, :, :, :].unsqueeze(0)).item())
                train_loss_count.update(loss.item(), label_l.shape[0])

                t.set_postfix({
                    'loss': '{0:1.5f}'.format(train_loss_count.avg),
                    'ssim': '{0:1.5f}'.format(train_ssim_count.avg)
                })
                t.update(1)

        lr = scheduler.get_last_lr()[0]
        tensorboard_writer.add_scalar(tag='Train Epoch/loss', scalar_value=train_loss_count.avg, global_step=epoch)
        tensorboard_writer.add_scalar(tag='Train Epoch/ssim', scalar_value=train_ssim_count.avg, global_step=epoch)
        tensorboard_writer.add_scalar(tag='Train Epoch/lr', scalar_value=lr, global_step=epoch)
        scheduler.step()

        if epoch % parameters.val_step == 0:
            # 3.4 保存训练模型
            train_model.save_pkl(parameters.train_pkl_path, 'trained')

            val_loss_count.reset()
            val_ssim_count.reset()
            val_psnr_count.reset()
            val_mse_count.reset()
            val_mae_count.reset()
            val_pearsonr_count.reset()

            train_model.load_pkl(parameters.best_pkl_path, mode='best', load=False)
            train_model.eval()

            # 3.5 验证，保存图片，保存最佳模型
            with tqdm(total=len(val_dl), ascii=True) as t2:
                t2.set_description(('Val'))
                for step, (label_ab, label_l, input_l, ms_img_o) in enumerate(val_dl):
                    input_l = input_l.float().to(parameters.device)
                    label_l = label_l.float().to(parameters.device)
                    label_ab = label_ab.float().to(parameters.device)
                    label_lab = torch.cat([label_l, label_ab], dim=1)
                    with torch.no_grad():
                        out_l, out_ab = train_model(input_l)
                        out_lab = torch.cat((out_l, out_ab), dim=1)
                    out_lab_cpu = out_lab.cpu().data.numpy()

                    if parameters.normalization:
                        out_lab_np = np.clip(out_lab_cpu[0] * 255, 0, 255).transpose((1, 2, 0)).astype(np.uint8)
                        img_lab = np.clip(label_lab.cpu().data.numpy()[0] * 255, 0, 255).transpose((1, 2, 0)).astype(
                            np.uint8)
                    else:
                        out_lab_np = np.clip(out_lab_cpu[0], 0, 255).transpose((1, 2, 0)).astype(np.uint8)
                        img_lab = np.clip(label_lab.cpu().data.numpy()[0], 0, 255).transpose((1, 2, 0)).astype(
                            np.uint8)
                    out_bgr = cv2.cvtColor(out_lab_np, cv2.COLOR_LAB2BGR)
                    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

                    # 保存10张图片
                    if step < 10:
                        cv2.imwrite(parameters.process_path + 'val' + str(epoch) + '_' + str(step).zfill(3) + '.png',
                                    np.hstack([img_bgr, ms_img_o[0], out_bgr]))
                    (psnr, ssim, mse, mae, pearsonr_corr) = fjn_util.cal_all_index(img_bgr, out_bgr)

                    val_loss = l1_loss_fun(out_l, label_l) + l1_loss_fun(out_ab, label_ab) - ssim_fun(out_lab,
                                                                                                      label_lab)
                    val_loss_count.update(val_loss, label_l.shape[0])
                    val_ssim_count.update(ssim)
                    val_psnr_count.update(psnr)
                    val_mse_count.update(mse)
                    val_mae_count.update(mae)
                    val_pearsonr_count.update(pearsonr_corr)

                    t2.set_postfix({
                        'psnr': '{0:1.5f}'.format(val_psnr_count.avg),
                        'ssim': '{0:1.5f}'.format(val_ssim_count.avg),
                        'mse': '{0:1.5f}'.format(val_mse_count.avg),
                        'mae': '{0:1.5f}'.format(val_mae_count.avg),
                        'loss': '{0:1.5f}'.format(val_loss_count.avg),
                        'pearsonr': '{0:1.5f}'.format(val_pearsonr_count.avg)
                    })
                    t2.update(1)

                # 保存最佳模型
                if train_model.best_psnr < val_psnr_count.avg:
                    train_model.best_epoch = epoch
                    train_model.best_psnr = val_psnr_count.avg
                    train_model.best_ssim = val_ssim_count.avg
                    train_model.save_pkl(parameters.best_pkl_path, 'best')
            tensorboard_writer.add_scalar(tag='Val/loss', scalar_value=val_loss_count.avg, global_step=epoch)
            tensorboard_writer.add_scalar(tag='Val/ssim', scalar_value=val_ssim_count.avg, global_step=epoch)
            tensorboard_writer.add_scalar(tag='Val/psnr', scalar_value=val_psnr_count.avg, global_step=epoch)
        train_model.train_epoch += 1


def test(parameters):
    statistic = fjn_util.Model_Statistics(statistics_path=parameters.test_statistics_path)
    # 4.1  加载测试数据集
    test_dl = Data_Loader(pan_src=parameters.img_test_pan_path,
                          ms_src=parameters.img_test_ms_path, batch_size=1,
                          shuffle=False, rand_rotate=False, rand_flip=False,
                          normalzero2one=parameters.normalization).load()

    # 4.2  加载最佳模型
    test_model = model.Net(parameters.device).to(parameters.device)
    # test_model.load_pkl(parameters.train_pkl_path,mode='trained',load=True)
    test_model.load_pkl(parameters.best_pkl_path, mode='best', load=True)
    test_model.eval()

    # 4.3  指标计算,保存图片
    test_psnr_count = fjn_util.AverageCounter()
    test_ssim_count = fjn_util.AverageCounter()
    test_mae_count = fjn_util.AverageCounter()
    test_mse_count = fjn_util.AverageCounter()
    test_crro_count = fjn_util.AverageCounter()
    test_sam_count = fjn_util.AverageCounter()
    test_cc_count = fjn_util.AverageCounter()
    test_lpips_count = fjn_util.AverageCounter()
    test_scc_count = fjn_util.AverageCounter()
    test_uiqi_count = fjn_util.AverageCounter()
    test_rmse_count = fjn_util.AverageCounter()
    test_ergas_count = fjn_util.AverageCounter()
    test_qnr_count = fjn_util.AverageCounter()
    test_tenengrad_count = fjn_util.AverageCounter()
    test_brenner_count = fjn_util.AverageCounter()
    test_laplacian_count = fjn_util.AverageCounter()
    test_smd_count = fjn_util.AverageCounter()
    test_smd2_count = fjn_util.AverageCounter()
    test_variance_count = fjn_util.AverageCounter()
    test_energy_count = fjn_util.AverageCounter()
    test_vollath_count = fjn_util.AverageCounter()
    test_entropy_count = fjn_util.AverageCounter()
    test_nrss_count = fjn_util.AverageCounter()
    with tqdm(total=len(test_dl), ascii=True) as t:
        for step, (label_ab, label_l, input_l, ms_img_o) in enumerate(test_dl):
            input_pan = label_l
            input_l = input_l.float().to(parameters.device)
            label_l = label_l.float().to(parameters.device)
            label_ab = label_ab.float().to(parameters.device)
            label_lab = torch.cat([label_l, label_ab], dim=1)

            with torch.no_grad():
                # torch.cuda.synchronize()
                # start = time.time()
                out_l, out_ab = test_model(input_l)
                # torch.cuda.synchronize()
                # end = time.time()
                # print("My", end - start)
                # exit(0)

                out = torch.cat((out_l, out_ab), dim=1)

            out_lab_cpu = out[-1].cpu().data.numpy()
            if parameters.normalization:
                out_lab = np.clip(out_lab_cpu * 255, 0, 255).transpose((1, 2, 0)).astype(np.uint8)
                img_lab = np.clip(label_lab.cpu().data.numpy()[0] * 255, 0, 255).transpose((1, 2, 0)).astype(np.uint8)
            else:
                out_lab = np.clip(out_lab_cpu[0], 0, 255).transpose((1, 2, 0)).astype(np.uint8)
                img_lab = np.clip(label_lab.cpu().data.numpy()[0], 0, 255).transpose((1, 2, 0)).astype(np.uint8)

            out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
            img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite(parameters.test_result_path + 'test' + '_' + str(step).zfill(3) + '.png', np.hstack([img_bgr,ms_img_o[0],out_bgr]))
            # if step in (0,69,85,116,117,140,158,189,195,200,201,221,138,121,169):
            #     cv2.imwrite(parameters.test_out_result_path + 'test' + '_' + str(step).zfill(3) + '.png', out_bgr)

            (psnr, ssim, mse, mae, pearsonr_corr, lpips) = fjn_util.cal_all_index(img_bgr, out_bgr)

            # if step in (190, 136, 162, 134, 90, 46, 138):
            cv2.imwrite(parameters.test_out_result_path + 'test' + '_' + str(step).zfill(3) + '(' + str(psnr) + ',' + str(
                        ssim) + ').png', out_bgr)

            (sam, cc, scc, uiqi, rmse, ergas) = iqa.cal_all_fusioneval_r(img_lab, out_lab)
            ms_lr = cv2.resize(img_bgr, (int(out_bgr.shape[0] / 4), int(out_bgr.shape[1] / 4)), cv2.INTER_CUBIC)
            qnr = iqa.cal_all_fusioneval_nr(out_bgr, np.squeeze(input_pan.cpu().data[0].numpy()), ms_lr)
            (tenengrad, brenner, laplacian, smd, smd2, variance, energy, vollath, entropy, nrss) = iqa.cal_all_nriq(
                cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY))
            test_psnr_count.update(psnr)
            test_ssim_count.update(ssim)
            test_mae_count.update(mae)
            test_mse_count.update(mse)
            test_lpips_count.update(lpips)
            test_crro_count.update(pearsonr_corr)

            test_sam_count.update(sam)
            test_cc_count.update(cc)
            test_scc_count.update(scc)
            test_uiqi_count.update(uiqi)
            test_rmse_count.update(rmse)
            test_ergas_count.update(ergas)

            test_qnr_count.update(qnr)

            test_tenengrad_count.update(tenengrad)
            test_brenner_count.update(brenner)
            test_laplacian_count.update(laplacian)
            test_smd_count.update(smd)
            test_smd2_count.update(smd2)
            test_variance_count.update(variance)
            test_energy_count.update(energy)
            test_vollath_count.update(vollath)
            test_entropy_count.update(entropy)
            test_nrss_count.update(nrss)
            statistic.update('val_psnr', test_psnr_count.avg)
            statistic.update('val_ssim', test_ssim_count.avg)
            statistic.write_all_to_txt()
            t.set_postfix({
                'psnr': '{0:1.5f}'.format(test_psnr_count.avg),
                'ssim': '{0:1.5f}'.format(test_ssim_count.avg),
                'mse': '{0:1.5f}'.format(test_mse_count.avg),
                'mae': '{0:1.5f}'.format(test_mae_count.avg),
                'pearsonr_corr': '{0:1.5f}'.format(test_crro_count.avg),
            })
            t.update(1)

    # 写入excel
    value_list = [test_model.best_epoch, test_psnr_count.avg, test_ssim_count.avg, test_mse_count.avg,
                  test_mae_count.avg, test_crro_count.avg, test_sam_count.avg, test_cc_count.avg, test_scc_count.avg,
                  test_uiqi_count.avg, test_rmse_count.avg, test_ergas_count.avg, test_qnr_count.avg,
                  test_tenengrad_count.avg,
                  test_brenner_count.avg, test_laplacian_count.avg, test_smd_count.avg, test_smd2_count.avg,
                  test_variance_count.avg, test_energy_count.avg, test_vollath_count.avg, test_entropy_count.avg,
                  test_nrss_count.avg]
    excel_write(begin_row=2, begin_line=11, value_list=value_list)


def loss_sum(result, label, weight):
    '''
    :param result: 特征图数组
    :param label: 标签
    :param weight: 权重数组（从小到大）
    :return: loss
    '''
    assert len(result) == len(weight)
    l1_loss_fun = nn.L1Loss()
    loss = 0
    for step, item in enumerate(result):
        loss += l1_loss_fun(item, label) * weight[step]
    return loss


class Data_Loader():
    def __init__(self, pan_src, ms_src, batch_size, shuffle=True, rand_rotate=False, rand_flip=False,
                 normalzero2one=False):
        self.dataset = Data_Set(pan_src, ms_src, rand_rotate, rand_flip, normalzero2one)
        self.batch = batch_size
        self.shuf = shuffle

        pass

    def load(self):
        loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch, shuffle=self.shuf,
                                             num_workers=0, drop_last=True)  # windows下设为0
        return loader
        # TODO
        # if self.sample:
        #     assert len(self.dataset) > self.train_num, 'train_num >= dataset length'
        #     # print(range(len(self.dataset))[:self.train_num])
        #     # print(range(len(self.dataset))[self.train_num:])
        #     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(len(self.dataset))[:self.train_num])
        #     val_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(len(self.dataset))[self.train_num:])
        #     train_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch, sampler=train_sampler,shuffle=self.shuf, num_workers=1, drop_last=True)
        #     val_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1, sampler=val_sampler,shuffle=self.shuf, num_workers=1, drop_last=True)
        #     return train_loader,val_loader
        # else:
        #     test_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=1,shuffle=self.shuf, num_workers=1, drop_last=True)
        #     return test_loader


class Data_Set(Dataset):
    def __init__(self, pan_src, ms_src, rand_rotate=False, rand_flip=False, normalzero2one=False, center_zero=False):
        self.pan_src = pan_src
        self.ms_src = ms_src
        self.src_list = os.listdir(pan_src)
        self.src_list.sort()
        self.rand_rotate = rand_rotate
        self.rand_flip = rand_flip
        self.normalzero2one = normalzero2one
        self.center_zero = center_zero

    def __getitem__(self, index):

        # pan_img: 256 × 256，ms_img: 64 × 64
        ms_img = cv2.imread(os.path.join(self.ms_src, self.src_list[index]), cv2.IMREAD_UNCHANGED)
        pan_img = cv2.imread(os.path.join(self.pan_src, self.src_list[index]), cv2.IMREAD_GRAYSCALE)

        if self.rand_rotate:
            # randint[], randrange[)
            rotate = random.randint(0, 3)
            rotate_arr = [0, 90, 180, 270]

            # getRotationMatrix2D：旋转中心，旋转角度，缩放比例
            matRotate_pan = cv2.getRotationMatrix2D((pan_img.shape[1] // 2, pan_img.shape[0] // 2), rotate_arr[rotate],
                                                    1)  # 旋转变化矩阵
            # warpAffine：输入图片，仿射变化矩阵，变换后的图片大小
            pan_img = cv2.warpAffine(pan_img, matRotate_pan, (pan_img.shape[1], pan_img.shape[0]))  # 旋转

            matRotate_ms = cv2.getRotationMatrix2D((ms_img.shape[1] // 2, ms_img.shape[0] // 2), rotate_arr[rotate],
                                                   1)  # 旋转变化矩阵
            ms_img = cv2.warpAffine(ms_img, matRotate_ms, (ms_img.shape[1], ms_img.shape[0]))  # 旋转

        if self.rand_flip:
            is_flip = random.randint(0, 1)
            if is_flip:
                flip = random.randint(-1, 1)
                # flip：输入图像，参数（>0 水平翻转，=0 垂直翻转，<0 水平和垂直翻转），输出图像（可选项）
                pan_img = cv2.flip(pan_img, flip)
                ms_img = cv2.flip(ms_img, flip)

        # pan_img: 256 × 256，ms_img: 64 × 64
        # label_l: 256 × 256，label_ab: 256 × 256
        # input_l: 128 × 128
        label_l = pan_img[np.newaxis, :]

        ms_img_o = cv2.resize(ms_img, (256, 256), interpolation=cv2.INTER_CUBIC)
        ms_lab = cv2.cvtColor(ms_img_o, cv2.COLOR_BGR2LAB)
        label_ab = ms_lab[:, :, 1:]

        pan_l_lr = cv2.resize(pan_img, (128, 128), interpolation=cv2.INTER_CUBIC)
        input_l = pan_l_lr[np.newaxis, :]

        if self.normalzero2one:
            label_ab = label_ab / 255.
            label_l = label_l / 255.
            input_l = input_l / 255.

        if self.center_zero:
            input_l = input_l - np.mean(input_l)

        # label_ab (高,宽,通道)->(通道,高,宽)
        label_ab = torch.from_numpy(label_ab.transpose((2, 0, 1)))  # 2, 256, 256
        input_l = input_l.repeat(3, axis=0)
        # ms_img_o: 上采样的 MS 图像
        return label_ab, label_l, input_l, ms_img_o

    def __len__(self):
        return len(self.src_list)


if __name__ == "__main__":
    parameters = set_parameters()

    # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    cudnn.benchmark = True

    # 为当前 GPU 设置种子用于生成随机数(确保每次生成固定的随机数)
    torch.cuda.manual_seed_all(parameters.seed)
    # 为当前 CPU 设置种子用于生成随机数(确保每次生成固定的随机数)
    torch.manual_seed(parameters.seed)
    # 每次返回的卷积算法将是确定的，即默认算法
    torch.backends.cudnn.deterministic = True
    print(parameters.train_pkl_path)

    fjn_util.make_folder(parameters.train_pkl_path)
    fjn_util.make_folder(parameters.best_pkl_path)
    fjn_util.make_folder(parameters.process_path)
    fjn_util.make_folder(parameters.test_result_path)
    fjn_util.make_folder(parameters.test_out_result_path)
    fjn_util.make_folder(parameters.test_statistics_path)
    # train(parameters)
    test(parameters)
