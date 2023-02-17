import os
import sys
import logging
from torch.autograd import Variable
from torch import nn,optim
from PIL import Image
import torch
from torch.utils.data import DataLoader
import numpy as np
from data import *
#from net import *
from torchvision.utils import save_image
import utils
from utils import MemoryFriendlyLoader

from UNet import Network


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'EXP//model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = 'EXP//image_epochs/'
os.makedirs(image_path, exist_ok=True)

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

if __name__ == '__main__':
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)


    # model = UNet().to(device)
    model = Network().to(device)


    opt = optim.Adam(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(),lr=0.003, betas=(0.9, 0.999), weight_decay=3e-4)
    loss_fun = nn.MSELoss

    # train_low_data_names = './LOLdataset/our485/low/'
    train_low_data_names = './results/low2/'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    # test_low_data_names = './LOLdataset/eval15/low/'
    test_low_data_names = './results/low2-e/'
    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False)

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False)

    epoch = 10
    total_step = 0
    for epoch in range(epoch):
        model.train()
        losses = []
        for batch_idx, (input, _) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()

            optimizer.zero_grad()
            # loss = nn.MSELoss(input)
            loss = model._loss(input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            # logging.info('train-epoch %03d %03d %f', epoch, batch_idx, loss)
            print('0 train-epoch {0},{1},{2}'.format(epoch, batch_idx, loss))

        # logging.info('train-epoch %03d %f', epoch, np.average(losses))
        print('1 train-epoch {0},{1}'.format(epoch, np.average(losses)))
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

        # if epoch % 1 == 0 and total_step != 0:
        # logging.info('train %03d %f', epoch, loss)
        print('2 train {0},{1}'.format(epoch, loss))
        model.eval()
        with torch.no_grad():
            for _, (input, image_name) in enumerate(test_queue):
                # input = Variable(input, volatile=True).cuda()
                with torch.no_grad():
                    input = Variable(input)
                    input = input.cuda()
                image_name = image_name[0].split('\\')[-1].split('.')[0]
                illu_list, ref_list, input_list = model(input)
                u_name = '%s.png' % (image_name + '_' + str(epoch))
                u_path = image_path + '/' + u_name
                # print("ref_list length：" + str(len(ref_list))+"illu_list length:"+str(len(illu_list)) + "input_list length:" + str(len(input_list)))
                save_images(ref_list[0], u_path)
        print("over!")