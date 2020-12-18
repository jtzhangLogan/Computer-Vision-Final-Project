"""
Our implementation
"""

import itertools
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import utils
from architecture import define_Gen, define_Dis, set_grad
from dataset import SurgicalVisDomDataset as Dataset
import tqdm


class enhanced_cycleGAN(object):
    def __init__(self,args):

        # Define the network
        self.Gxy = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gyx = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Dx = define_Dis(input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Dx_tmp = define_Dis(input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Dy = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Dy_tmp = define_Dis(input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gxy, self.Gyx, self.Dx, self.Dy], ['Gab', 'Gba', 'Da', 'Db'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Optimizers
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gxy.parameters(),self.Gyx.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Dx.parameters(),self.Dy.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        self.X_fake_sample = utils.Sample_from_Pool()
        self.Y_fake_sample = utils.Sample_from_Pool()

        # Try loading checkpoint
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Dx.load_state_dict(ckpt['Dx'])
            self.Dy.load_state_dict(ckpt['Dy'])
            self.Gxy.load_state_dict(ckpt['Gxy'])
            self.Gyx.load_state_dict(ckpt['Gyx'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train_G(self, X, Y, round, args):
        # Generator
        # X -> Y
        X2Y_1 = self.Gxy(X[round])
        X2Y_2 = self.Gxy(X[round + 1])

        # Y -> X
        Y2X_1 = self.Gyx(Y[round])
        Y2X_2 = self.Gyx(Y[round + 1])

        # Cycle
        X2Y2X_1 = self.Gyx(X2Y_1)
        X2Y2X_2 = self.Gyx(X2Y_2)
        Y2X2Y_1 = self.Gxy(Y2X_1)
        Y2X2Y_2 = self.Gxy(Y2X_2)

        # Identity
        X2X_1 = self.Gyx(X[round])
        X2X_2 = self.Gyx(X[round + 1])
        Y2Y_1 = self.Gxy(Y[round])
        Y2Y_2 = self.Gxy(Y[round + 1])

        # Adversarial losses
        X2Y_2_dis = self.Dy(X2Y_2)
        X2Y_2_real_label = Variable(torch.ones(X2Y_2_dis.size())).cuda()
        X_gen_loss = self.MSE(X2Y_2_dis, X2Y_2_real_label)
        Y2X_2_dis = self.Dx(Y2X_2)
        Y2X_2_real_label = Variable(torch.ones(Y2X_2_dis.size())).cuda()
        Y_gen_loss = self.MSE(Y2X_2_dis, Y2X_2_real_label)

        # Identity losses
        X_idt_loss = (self.L1(X2X_1, X[round]) + self.L1(X2X_2, X[round + 1])) * args.lamda * args.idt_coef
        Y_idt_loss = (self.L1(Y2Y_1, Y[round]) + self.L1(Y2Y_2, Y[round + 1])) * args.lamda * args.idt_coef

        # Cycle consistency losses
        X_cycle_loss = self.L1(X2Y2X_1, X[round]) * args.lamda + self.L1(X2Y2X_2, X[round + 1]) * args.lamda
        Y_cycle_loss = self.L1(Y2X2Y_1, Y[round]) * args.lamda + self.L1(Y2X2Y_2, Y[round + 1]) * args.lamda

        # Total generators losses
        gen_loss = X_gen_loss + Y_gen_loss + X_cycle_loss + Y_cycle_loss + X_idt_loss + Y_idt_loss

        # Update generators
        gen_loss.backward()
        self.g_optimizer.step()

        return X2Y_2, Y2X_2, gen_loss

    def train_D(self, X, Y, X2Y, Y2X, round):
        # Dx
        X_real_dis = self.Dx(X[round + 1])
        X_fake_dis = self.Dx(Y2X)
        X_real_label = Variable(torch.ones(X_real_dis.size())).cuda()
        X_fake_label = Variable(torch.zeros(X_fake_dis.size())).cuda()

        # Discriminator losses
        X_dis_real_loss = self.MSE(X_real_dis, X_real_label)
        X_dis_fake_loss = self.MSE(X_fake_dis, X_fake_label)
        X_dis_loss = (X_dis_real_loss + X_dis_fake_loss) * 0.5

        # Dy
        Y_real_dis = self.Dy(Y[round + 1])
        Y_fake_dis = self.Dy(X2Y)
        Y_real_label = Variable(torch.ones(Y_real_dis.size())).cuda()
        Y_fake_label = Variable(torch.zeros(Y_fake_dis.size())).cuda()

        # Discriminator losses
        Y_dis_real_loss = self.MSE(Y_real_dis, Y_real_label)
        Y_dis_fake_loss = self.MSE(Y_fake_dis, Y_fake_label)
        Y_dis_loss = (Y_dis_real_loss + Y_dis_fake_loss) * 0.5

        # Total discriminators losses
        total_loss = X_dis_loss + Y_dis_loss

        # Update discriminators
        X_dis_loss.backward()
        Y_dis_loss.backward()
        self.d_optimizer.step()

        return total_loss

    def train_round(self, X, Y, args, round):
        # Generator Computations
        set_grad([self.Dx, self.Dy], False)
        self.g_optimizer.zero_grad()
        X2Y, Y2X, G_loss = self.train_G(X, Y, round=round, args=args)

        # Sample from history of generated images
        X2Y = Variable(torch.Tensor(self.Y_fake_sample([X2Y.cpu().data.numpy()])[0])).cuda()
        Y2X = Variable(torch.Tensor(self.X_fake_sample([Y2X.cpu().data.numpy()])[0])).cuda()

        # Discriminator Computations
        set_grad([self.Dx, self.Dy], True)
        self.d_optimizer.zero_grad()
        D_loss = self.train_D(X, Y, X2Y, Y2X, round)

        return G_loss, D_loss

    def train(self, args):
        # For transforming the input image
        transform = transforms.Compose(
            [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Load train data
        train = Dataset('./input_data', "train", transform=transform)
        loader_train = DataLoader(
            train,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
        )
        loaders = {"train": loader_train}

        for epoch in tqdm.tqdm(range(self.start_epoch, args.epochs)):
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for m, data in enumerate(loaders["train"]):
                vr, surgery = data
                X = []
                Y = []

                for i in range(len(vr)):
                    X.append(Variable(vr[i]).cuda())
                for i in range(len(surgery)):
                    Y.append(Variable(surgery[i]).cuda())

                G_loss_round_0, D_loss_round_0 = self.train_round(X, Y, args, round=0)
                G_loss_round_1, D_loss_round_1 = self.train_round(X, Y, args, round=1)

                if m % 100 == 0:
                    print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                (epoch + 1, m + 1, len(train), (G_loss_round_0 + G_loss_round_1) / 2, (D_loss_round_0 + D_loss_round_1) / 2))

                if m == len(train) - 3:
                    break

            # Override the latest checkpoint
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Dx': self.Dx.state_dict(),
                                   'Dy': self.Dy.state_dict(),
                                   'Gxy': self.Gxy.state_dict(),
                                   'Gyx': self.Gyx.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                   '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()


class cycleGAN(object):
    def __init__(self,args):

        # Define the network
        self.Gxy = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gyx = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Dx = define_Dis(input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Dy = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gxy, self.Gyx, self.Dx, self.Dy], ['Gab', 'Gba', 'Da', 'Db'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Optimizers
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gxy.parameters(),self.Gyx.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Dx.parameters(),self.Dy.parameters()), lr=args.lr, betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        self.X_fake_sample = utils.Sample_from_Pool()
        self.Y_fake_sample = utils.Sample_from_Pool()

        # Try loading checkpoint
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Dx.load_state_dict(ckpt['Dx'])
            self.Dy.load_state_dict(ckpt['Dy'])
            self.Gxy.load_state_dict(ckpt['Gxy'])
            self.Gyx.load_state_dict(ckpt['Gyx'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train_G(self, X, Y, args):
        # Generator
        # X -> Y
        X2Y = self.Gxy(X[0])

        # Y -> X
        Y2X = self.Gyx(Y[0])

        # Cycle
        X2Y2X = self.Gyx(X2Y)
        Y2X2Y = self.Gxy(Y2X)

        # Identity
        X2X = self.Gyx(X[0])
        Y2Y = self.Gxy(Y[0])

        # Adversarial losses
        X2Y_dis = self.Dy(X2Y)
        X2Y_real_label = Variable(torch.ones(X2Y_dis.size())).cuda()
        X_gen_loss = self.MSE(X2Y_dis, X2Y_real_label)
        Y2X_dis = self.Dx(Y2X)
        Y2X_real_label = Variable(torch.ones(Y2X_dis.size())).cuda()
        Y_gen_loss = self.MSE(Y2X_dis, Y2X_real_label)

        # Identity losses
        X_idt_loss = self.L1(X2X, X[0]) * args.lamda * args.idt_coef
        Y_idt_loss = self.L1(Y2Y, Y[0]) * args.lamda * args.idt_coef

        # Cycle consistency losses
        X_cycle_loss = self.L1(X2Y2X, X[0]) * args.lamda
        Y_cycle_loss = self.L1(Y2X2Y, Y[0]) * args.lamda

        # Total generators losses
        gen_loss = X_gen_loss + Y_gen_loss + X_cycle_loss + Y_cycle_loss + X_idt_loss + Y_idt_loss

        # Update generators
        gen_loss.backward()
        self.g_optimizer.step()

        return X2Y, Y2X, gen_loss

    def train_D(self, X, Y, X2Y, Y2X):
        # Dx
        X_real_dis = self.Dx(X[0])
        X_fake_dis = self.Dx(Y2X)
        X_real_label = Variable(torch.ones(X_real_dis.size())).cuda()
        X_fake_label = Variable(torch.zeros(X_fake_dis.size())).cuda()

        # Discriminator losses
        X_dis_real_loss = self.MSE(X_real_dis, X_real_label)
        X_dis_fake_loss = self.MSE(X_fake_dis, X_fake_label)
        X_dis_loss = (X_dis_real_loss + X_dis_fake_loss) * 0.5

        # Dy
        Y_real_dis = self.Dy(Y[0])
        Y_fake_dis = self.Dy(X2Y)
        Y_real_label = Variable(torch.ones(Y_real_dis.size())).cuda()
        Y_fake_label = Variable(torch.zeros(Y_fake_dis.size())).cuda()

        # Discriminator losses
        Y_dis_real_loss = self.MSE(Y_real_dis, Y_real_label)
        Y_dis_fake_loss = self.MSE(Y_fake_dis, Y_fake_label)
        Y_dis_loss = (Y_dis_real_loss + Y_dis_fake_loss) * 0.5

        # Total discriminators losses
        total_loss = X_dis_loss + Y_dis_loss

        # Update discriminators
        X_dis_loss.backward()
        Y_dis_loss.backward()
        self.d_optimizer.step()

        return total_loss

    def train_round(self, X, Y, args):
        # Generator Computations
        set_grad([self.Dx, self.Dy], False)
        self.g_optimizer.zero_grad()
        X2Y, Y2X, G_loss = self.train_G(X, Y, args=args)

        # Sample from history of generated images
        X2Y = Variable(torch.Tensor(self.Y_fake_sample([X2Y.cpu().data.numpy()])[0])).cuda()
        Y2X = Variable(torch.Tensor(self.X_fake_sample([Y2X.cpu().data.numpy()])[0])).cuda()

        # Discriminator Computations
        set_grad([self.Dx, self.Dy], True)
        self.d_optimizer.zero_grad()
        D_loss = self.train_D(X, Y, X2Y, Y2X)

        return G_loss, D_loss

    def train(self, args):
        # For transforming the input image
        transform = transforms.Compose(
            [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # Load train data
        train = Dataset('./input_data', "train", transform=transform)
        loader_train = DataLoader(
            train,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=4,
        )
        loaders = {"train": loader_train}

        for epoch in tqdm.tqdm(range(self.start_epoch, args.epochs)):
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for m, data in enumerate(loaders["train"]):
                vr, surgery = data
                X = []
                Y = []

                for i in range(len(vr)):
                    X.append(Variable(vr[i]).cuda())
                for i in range(len(surgery)):
                    Y.append(Variable(surgery[i]).cuda())

                G_loss, D_loss = self.train_round(X, Y, args)

                if m % 100 == 0:
                    print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                (epoch + 1, m + 1, len(train), G_loss, D_loss))

                if m == len(train) - 3:
                    break

            # Override the latest checkpoint
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Dx': self.Dx.state_dict(),
                                   'Dy': self.Dy.state_dict(),
                                   'Gxy': self.Gxy.state_dict(),
                                   'Gyx': self.Gyx.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                   '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()