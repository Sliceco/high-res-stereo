import torch
from dataloader import listfiles as ls
from dataloader import listsceneflow as lt
from dataloader import KITTIloader2015 as lk15
from dataloader import KITTIloader2012 as lk12
from dataloader import MiddleburyLoader as DA
from dataloader import lidar_loader as lld


def get_training_dataloader(maxdisp, dataset_folder):
    scale_factor = maxdisp / 384.

    all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader(
        '%s/hrvs/carla-highres/trainingF' % dataset_folder)
    loader_carla = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                    rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=2)

    all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader(
        '%s/middlebury/mb-ex-training/trainingF' % dataset_folder)  # mb-ex
    loader_mb = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                 rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=0)

    all_left_img, all_right_img, all_left_disp, all_right_disp = lt.dataloader('%s/sceneflow/' % dataset_folder)
    loader_scene = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                    rand_scale=[0.9, 2.4 * scale_factor], order=2)

    all_left_img, all_right_img, all_left_disp, _, _, _ = lk15.dataloader('%s/kitti15/training/' % dataset_folder,
                                                                          typ='train')  # change to trainval when finetuning on KITTI
    loader_kitti15 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.9, 2.4 * scale_factor],
                                      order=0)
    all_left_img, all_right_img, all_left_disp = lk12.dataloader('%s/kitti12/training/' % dataset_folder)
    loader_kitti12 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.9, 2.4 * scale_factor],
                                      order=0)

    all_left_img, all_right_img, all_left_disp, _ = ls.dataloader('%s/eth3d/' % dataset_folder)
    loader_eth3d = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.9, 2.4 * scale_factor],
                                    order=0)

    all_left_img, all_right_img, all_left_disp = lld.dataloader('%s/lidar_dataset/train' % dataset_folder)
    loader_lidar = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.5, 1.25 * scale_factor],
                                    rand_bright=[0.8, 1.2], order=0)
    all_dataloaders = [{'name': 'lidar', 'dl': loader_lidar, 'count': 1},
                       {'name': 'hrvs', 'dl': loader_carla, 'count': 1},
                       {'name': 'middlebury', 'dl': loader_mb, 'count': 1},
                       {'name': 'sceneflow', 'dl': loader_scene, 'count': 1},
                       {'name': 'kitti12', 'dl': loader_kitti12, 'count': 1},
                       {'name': 'kitti15', 'dl': loader_kitti15, 'count': 1},
                       {'name': 'eth3d', 'dl': loader_eth3d, 'count': 1}]
    max_count = 0
    for dataloader in all_dataloaders:
        max_count = max(max_count, len(dataloader['dl']))

    print('=' * 80)
    concat_dataloaders = []
    for dataloader in all_dataloaders:
        dataloader['count'] = max(1, max_count // len(dataloader['dl']))
        concat_dataloaders += [dataloader['dl']] * dataloader['count']
        print('{name}: {size} (x{count})'.format(name=dataloader['name'],
                                                 size=len(dataloader['dl']),
                                                 count=dataloader['count']))
    data_inuse = torch.utils.data.ConcatDataset(concat_dataloaders)
    print('Total dataset size: {}'.format(len(data_inuse)))
    print('=' * 80)
    return data_inuse
