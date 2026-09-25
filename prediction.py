import torch
import argparse
import numpy as np
from utils.metrics import eval
from torch.utils.data import DataLoader
from dataloader.dataset import build_Dataset
from dataloader.transforms import build_transforms




def test():
    from model.model_Jump import Semi_GMS

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='D:/Paper Engineering Data/2D_data',
                        help='Name of Experiment')
    
    # parser.add_argument('--dataset', type=str, default='/BUSI',
    #                     help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/BCSS',
    #                     help='Name of Experiment')
    parser.add_argument('--dataset', type=str, default='/tumor_10',
                        help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/thyroid_10',
    #                     help='Name of Experiment')
    # parser.add_argument('--dataset', type=str, default='/ISIC_TrainDataset_10',
    #                 help='Name of Experiment')

    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='input channel of network')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')

    parser.add_argument('--model_path', type=str,
                        default="./Results/result_tumor_30/fold_0/best_model_434.pth",
                        help='model weight path')

    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    

    data_transforms = build_transforms(args)
    test_dataset_list = ["test_CVC-300", "test_CVC-ClinicDB", "test_CVC-ColonDB", "test_ETIS-LaribPolypDB", "test_Kvasir"]
    # test_dataset_list = ["test_CVC-300", "test_CVC-ClinicDB", "test_Kvasir"]
    # test_dataset_list = ["test_Kvasir"]
    # test_dataset_list = ["test_CVC-ColonDB", "test_ETIS-LaribPolypDB"]
    # test_dataset_list = ["test_DDTI", "test_tn3k"]
    # test_dataset_list = ["test_ISIC2018"]
    # test_dataset_list = ["test_BCSS"]
    # test_dataset_list = ["test_BUSI"]

    model = Semi_GMS(args=args, in_chns=args.in_channels, class_num=args.num_classes).cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    

    for test_dataset_name in test_dataset_list:
        test_dataset = build_Dataset(args=args, data_dir=args.data_path + args.dataset, split=test_dataset_name,
                                     transform=data_transforms["valid_test"])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
        
        mean_avg_dice_list = []
        mean_avg_iou_list = []
        mean_avg_hd95_list = []
        
        for i_batch, sampled_batch in enumerate(test_loader):
            test_image, test_label = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            if test_image.shape[1] == 1:
                test_image = test_image.repeat(1, 3, 1, 1)
                
            img_rgb = 2. * test_image - 1.
            vae_pred, cnn_pred  = model(img_rgb, None, is_train=False)
            
            mean_eval_list = eval(test_label, (vae_pred + cnn_pred) / 2, thr=0.5)


            
            mean_avg_dice_list.append(mean_eval_list[0])
            mean_avg_iou_list.append(mean_eval_list[1])
            mean_avg_hd95_list.append(mean_eval_list[2])
            
            
            # print(mean_eval_list[0])


        mean_avg_dice = np.mean(mean_avg_dice_list)
        mean_avg_hd95 = np.mean(mean_avg_hd95_list)
        mean_avg_iou = np.mean(mean_avg_iou_list)

        print(test_dataset_name, " :")
        print("mean_avg_dice: ", mean_avg_dice)
        print("mean_avg_iou: ", mean_avg_iou)
        print("mean_avg_hd95: ", mean_avg_hd95)
        
    # return avg_dice, avg_iou, avg_hd95



if __name__ == '__main__':
    test()

