import argparse
import logging
import core.logger as Logger
import data as Data

#Create chaneg detection dataset
import logging
import torch.utils.data

def create_cd_dataloader(dataset, dataset_opt, phase):
    if phase == 'train' or 'val' or 'test':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found'.format(phase)
        )

def create_cd_dataset(dataset_opt, phase):
    from data.CDDataset import CDDataset
    print(dataset_opt["datasetroot"])
    dataset = CDDataset(root_dir=dataset_opt["datasetroot"],
                        resolution=dataset_opt["resolution"],
                        split=phase,
                        data_len=dataset_opt["data_len"]
                        )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s} - {:s}] is created'.format(dataset.__class__.__name__,
                                                                 dataset_opt['name'],
                                                                 phase))
    return dataset

def create_scd_dataset(dataset_opt, phase):
    from data.CDDataset import SCDDataset
    print(dataset_opt["datasetroot"])
    dataset = SCDDataset(root_dir=dataset_opt["datasetroot"],
                        resolution=dataset_opt["resolution"],
                        split=phase,
                        data_len=dataset_opt["data_len"]
                        )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s} - {:s}] is created'.format(dataset.__class__.__name__,
                                                                 dataset_opt['name'],
                                                                 phase))
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config/levir.json')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    print(opt)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] change-detection dataloader.")
            train_set  = Data.create_cd_dataset(dataset_opt, phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)

