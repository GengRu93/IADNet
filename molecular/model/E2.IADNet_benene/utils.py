import numpy as np
import os, torch, pickle, zipfile
import imageio, shutil
import scipy, scipy.misc, scipy.integrate

solve_ivp = scipy.integrate.solve_ivp


def L2_loss(u, v):
    return (u - v).pow(2).mean()


def weights_init(m):
    if isinstance(m, torch.nn.Linear):  # 函数来判断一个对象是否是一个已知的类型
        torch.nn.init.xavier_normal_(m.weight.data, gain=1.66666667)
        torch.nn.init.zeros_(m.bias.data)


def get_parameters_count(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_params


def get_diff(u_reshaped, dt):
    u1 = u_reshaped[:-1, :]
    # print('u1.shape')
    # print(u1.shape)
    u2 = u_reshaped[1:, :]
    diff_u = (u2 - u1) / dt
    return u1, u2, diff_u


def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'silu':
        nl = torch.nn.functional.silu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    elif name == 'squareplus':
        nl = lambda x: 1/2*(x+torch.sqrt(x*x+4))
    else:
        raise ValueError("nonlinearity not recognized")
    return nl


def make_gif(frames, save_dir, name='pendulum', duration=1e-1, pixels=None, divider=0):
    '''Given a three dimensional array [frames, height, width], make
    a gif and save it.'''
    temp_dir = './_temp'
    os.mkdir(temp_dir) if not os.path.exists(temp_dir) else None
    for i in range(len(frames)):
        im = (frames[i].clip(-.5, .5) + .5) * 255
        im[divider, :] = 0
        im[divider + 1, :] = 255
        if pixels is not None:
            im = scipy.misc.imresize(im, pixels)
        scipy.misc.imsave(temp_dir + '/f_{:04d}.png'.format(i), im)

    images = []
    for file_name in sorted(os.listdir(temp_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(temp_dir, file_name)
            images.append(imageio.imread(file_path))
    save_path = '{}/{}.gif'.format(save_dir, name)
    png_save_path = '{}.png'.format(save_path)
    imageio.mimsave(save_path, images, duration=duration)
    os.rename(save_path, png_save_path)

    shutil.rmtree(temp_dir)  # remove all the images
    return png_save_path