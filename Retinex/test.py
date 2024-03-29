import os
import sys
import torch

import torchvision.datasets as datasets
from UNet import *
from torchvision.utils import save_image
from utils import MemoryFriendlyLoader, MemoryFriendlyLoader1

data_path = './results/low2-e'
save_path = './results/low3'
os.makedirs(save_path, exist_ok=True)

# net=UNet().cuda()

TestDataset = MemoryFriendlyLoader(img_dir=data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    model = Finetunemodel('./EXP/model_epochs/weights_9.pt')
    model = model.cuda()

    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('\\')[-1].split('.')[0]
            # input =transforms.ToTensor(input)
            # print(input.shape)
            i, r = model(input)
            u_name = '%s.png' % (image_name)
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            # input_op =i + r * input
            input_op = i*r*input

            save_images(input_op, u_path)



if __name__ == '__main__':
    main()