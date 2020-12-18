"""
Our implementation
"""

import os
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import utils
from architecture import define_Gen
from dataset import SurgicalVisDomDataset as Dataset
from torch.utils.data import DataLoader
import timeit


def test(args):
    transform = transforms.Compose(
        [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # Load train data
    test = Dataset('./input_data', "test", transform=transform)
    loader_train = DataLoader(
        test,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )
    loaders = {"test": loader_train}

    Gxy = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
    Gyx = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)  # resnet_6blocks  unet_256

    utils.print_networks([Gxy,Gyx], ['Gxy','Gyx'])

    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
    Gxy.load_state_dict(ckpt['Gxy'])
    Gyx.load_state_dict(ckpt['Gyx'])

    for i, data in enumerate(loaders["test"]):
        vr, surgery = data

        x_real_test = Variable(vr[0]).cuda()
        y_real_test = Variable(surgery[0]).cuda()

        Gxy.eval()
        Gyx.eval()

        with torch.no_grad():
            start = timeit.timeit()
            y_fake_test = Gxy(x_real_test)
            end = timeit.timeit()
            x_fake_test = Gyx(y_real_test)
            y_recon_test = Gxy(x_fake_test)
            x_recon_test = Gyx(y_fake_test)

        print("process time = {}".format(end-start))

        pic = (torch.cat([x_real_test, y_fake_test, x_recon_test, y_real_test, x_fake_test, y_recon_test], dim=0).data+ 1) / 2

        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        torchvision.utils.save_image(pic, args.results_dir+'/sample{}.jpg'.format(i), nrow=3)

