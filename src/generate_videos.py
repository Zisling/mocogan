"""
Usage:
    generate_videos.py --model_path=<path> --out_path=<path>

"""

import os
import torch

from trainers import videos_to_numpy

import docopt
import numpy as np
import imageio


def crate_gif(video, filename):
    imageio.mimsave(filename, video)


if __name__ == "__main__":
    print('Setting up video generating program')
    args = docopt.docopt(__doc__)
    print(args)

    model_path = args['--model_path']
    out_path = args['--out_path']
    generator = torch.load(model_path)
    print('Loaded model')
    generator.eval()
    print('Evaluated generator')
    num_videos = 10

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print('Starting generation')
    for i in range(num_videos):
        v, _ = generator.sample_videos(1, 35 * 2)
        video = videos_to_numpy(v).squeeze().transpose((1, 2, 3, 0))
        print(video.shape)
        crate_gif(video[:, :, :, 0:3], os.path.join(out_path, "{}_objs.{}".format(i, "gif")))
        crate_gif(video[:, :, :, 3:6], os.path.join(out_path, "{}_background.{}".format(i, "gif")))
        crate_gif(video[:, :, :, 6], os.path.join(out_path, "{}_depth.{}".format(i, "gif")))
        print('finished gif no. {}'.format(i))
    print('Finished generation')
