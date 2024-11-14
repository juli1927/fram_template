

import os
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("../spatial/")

import time
import argparse
import warnings
warnings.filterwarnings("ignore")

from colorama import Fore

from utils import *
from dataloader import *

from Models.Pix2Pix import *
# from Models.attention_unet import PA_UNet_Generator
from Models.pixel_attention import PAN_Generator
from Models.Dual_Pix2Pix import DualGeneratorUNet
from Models.TSGAN import TSG_Generator
from Models.EaGAN import create_eagan_model


def run_validation(args): 
    #
    import torch
    import torchvision.transforms as transforms

    if args.model == "UNet":
        # Initialize generator 
        generator = GeneratorUNet(n_channels = args.channels)
        # to_cuda = [generator] 

    elif args.model == "Pix2Pix":
        # Initialize generator 
        generator = GeneratorUNet(n_channels = args.channels)
        # to_cuda = [generator] 
    
    elif args.model == "DualPix2Pix":
        # Initialize generator 
        generator = DualGeneratorUNet(n_channels = args.channels)
        # to_cuda = [generator] 
    
    elif args.model == "PAN": 
        generator = PAN_Generator(in_nc = args.channels, out_nc = args.channels, nf=40, unf=24, nb=16, scale=1)
        # to_cuda = [generator] 
    
    else: 
        # Initialize generator 
        generator = GeneratorUNet(n_channels = args.channels)
        # to_cuda = [generator] 
    
    to_cuda = [generator] 

    # Move everything to gpu
    if args.cuda:
        for model in to_cuda:
            model.cuda()
    
    else:
        for model in to_cuda:
            model.cpu()

    # Load model in case of training. Otherwise, random init
    map_location = None if args.cuda else "cpu"
    generator.load_state_dict(torch.load("{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, args.epoch), map_location = map_location))
    print (Fore.GREEN + "Weights from checkpoint: {0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, args.epoch) + Fore.RESET)
    
    #    
    # Configure dataloaders
    transforms_ = [
        transforms.Resize((args.image_size, args.image_size), Image.BICUBIC),
        transforms.ToTensor(),
    ] # z_score(),

    roi_transforms_ = [
        transforms.Resize((args.roi_size, args.roi_size), Image.BICUBIC),
        transforms.ToTensor(),
    ] # z_score(),

    if args.normalization == "min_max":
        #
        transforms_.append(min_max_scaling(out_range = [-1,1]))
        roi_transforms_.append(min_max_scaling(out_range = [-1,1]))

    elif args.normalization == "z_score":
        #
        transforms_.append(z_score())
        roi_transforms_.append(z_score())

    elif args.normalization == "ti_norm":
        #
        transforms_.append(z_score())
        roi_transforms_.append(z_score())
    

    # Initialize data loader
    pre_early = Loader (input_sequence = args.input[0], output_sequence = args.output, data_path = args.data_dir, quality = args.quality,
                             batch_size = args.batch_size, 
                             img_res=(args.image_size, args.image_size), roi_size = (args.roi_size,args.roi_size), 
                             n_channels = args.channels,
                             train_transforms = transforms_, val_transforms = transforms_, roi_transforms = roi_transforms_, 
                             dataset_name = args.dataset_name, workers=args.num_workers,
                             norm=args.normalization)
    
    # Initialize data loader
    # early_late = Loader (input_sequence = args.output, output_sequence = args.input[1], data_path = args.data_dir, quality = args.quality,
    #                         batch_size = args.batch_size, 
    #                         img_res=(args.image_size, args.image_size), roi_size = (args.roi_size,args.roi_size), 
    #                         n_channels = args.channels,
    #                         train_transforms = transforms_, val_transforms = transforms_, roi_transforms = roi_transforms_, 
    #                         dataset_name = args.dataset_name, workers=args.num_workers,
    #                         norm=args.normalization)
    
    # Initialize email notifier 
    prev_time = None
    # notifier = setup_notifier(mode=args.notification_mode, send_notifications=not args.no_notifications)
    
    # In case of validation option. Otherwise, move to train
    if args.pixel_metrics: 
        #
        exp = args.exp_name[args.exp_name.rfind("/")+1:]
        
        output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch)

        print ("\n" + Fore.BLUE + "[*] -> Generating Pixel metrics.... " + Fore.RESET + "\n")

        subj = "📊   Generating Pixel metrics.... \n"
        body = "     - Model: {0} \n".format(args.model) + \
               "     - Normalization: {0} \n".format(args.normalization) + \
               "     - Exp name: {0} \n".format(args.exp_name)
        
        # notify(notifier, args.exp_name, subj, body, prev_time)
        prev_time = time.time()

        generate_images_with_stats( args, pre_early, generator, args.epoch, \
                                    shuffled = False, write_log = True, masked = True, image_level = True, \
                                    output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch), \
                                    exp = exp, save_file = output_dir + "px_stats.csv")
        
        print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")
        if not args.time_intensities and not args.random_patches_metrics: 
            subj = "✅   Pixel metrics Completed! \n"
            # notify(notifier, args.exp_name, subj, body, prev_time)
            exit()
    
    if args.time_intensities: #args.ce_metrics: 
        #
        exp = args.exp_name[args.exp_name.rfind("/")+1:]
        
        output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch)

        print ("\n" + Fore.BLUE + "[*] -> Generating Time intensities...." + Fore.RESET + "\n")

        subj = "📊   Generating Time intensities.... \n"
        body = "     - Model: {0} \n".format(args.model) + \
               "     - Normalization: {0} \n".format(args.normalization) + \
               "     - Exp name: {0} \n".format(args.exp_name)
        
        notify(notifier, args.exp_name, subj, body, prev_time)
        if prev_time == None: prev_time = time.time()

        generate_time_intensity_reference(args, pre_early, early_late, generator, args.epoch, \
                                            point_max=True, draw_patches = True, \
                                            output_dir = output_dir, save_file = output_dir + "im_ti_stats.csv")
        
        print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")
        if not args.time_intensities and not args.random_patches_metrics and \
           not args.random_patches_points and not args.mask_metrics: 
            subj = "✅   Time intensities Completed! \n"
            notify(notifier, args.exp_name, subj, body, prev_time)
            exit()
    
    if args.random_patches_metrics: 
        #
        assert args.random_points_patches != None, ValueError (Fore.RED + "Must provide a --random_points_patches file." + Fore.RESET)

        exp = args.exp_name[args.exp_name.rfind("/")+1:]
        
        output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch)

        print ("\n" + Fore.BLUE + "[*] -> Generating Patches metrics...." + Fore.RESET + "\n")

        subj = "📊   Generating Patches intensities.... \n"
        body =  "     - Model: {0} \n".format(args.model) + \
                "     - Normalization: {0} \n".format(args.normalization) + \
                "     - Exp name: {0} \n".format(args.exp_name) + \
                "     - Random patch size = {0}x{0} \n".format(args.random_roi_size)
        
        notify(notifier, args.exp_name, subj, body, prev_time)
        if prev_time == None: prev_time = time.time()
    
        generate_time_intensity_patches(args, pre_early, early_late, generator, args.epoch, \
                                        shuffle = False, point_max=True, draw_patches = True, \
                                        output_dir = output_dir, save_file = output_dir + "pt_{0}_ti_stats.csv".format(args.random_roi_size), \
                                        points = args.random_points_patches)
            
        print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")
        subj = "✅   Patches Metrics Completed! \n"
        notify(notifier, args.exp_name, subj, body, prev_time)
        
        if not args.random_patches_points and not args.mask_metrics: 
            subj = "✅   Random patches metrics Completed! \n"
            notify(notifier, args.exp_name, subj, body, prev_time)
            exit()
    
    if args.random_patches_points: 
        # 
        exp = args.exp_name[args.exp_name.rfind("/")+1:]

        print ("\n" + Fore.BLUE + "[*] -> Generating points for random patches...." + Fore.RESET + "\n")
        subj = "📊   Generating points for random patches.... \n"
        body = "     - Patch size = {0}x{0} \n".format(args.random_roi_size)
        notify(notifier, args.exp_name, subj, body, prev_time)
        if prev_time == None: prev_time = time.time()
    
        generate_time_intensity_points(args, pre_early, early_late, shuffle = False, \
                                        output_dir = args.result_dir, save_file = args.result_dir + "/generated_images/points_{0}x{0}.csv".format(args.random_roi_size))
            
        print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")
        subj = "✅   Points generated sucessfully! \n"
        notify(notifier, args.exp_name, subj, body, prev_time)

        if not args.mask_metrics: 
            subj = "✅   Random patches points generated! \n"
            notify(notifier, args.exp_name, subj, body, prev_time)
            exit()

    # In case of validation option. Otherwise, move to train
    if args.mask_metrics: 
        #
        exp = args.exp_name[args.exp_name.rfind("/")+1:]
        
        output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch)

        print ("\n" + Fore.BLUE + "[*] -> Generating Masks metrics.... " + Fore.RESET + "\n")

        subj = "📊   Generating Masks metrics.... \n"
        body = "     - Model: {0} \n".format(args.model) + \
                "     - Normalization: {0} \n".format(args.normalization) + \
                "     - Exp name: {0} \n".format(args.exp_name)
        
        notify(notifier, args.exp_name, subj, body, prev_time)
        prev_time = time.time()

        if args.model == 'DualPix2Pix': 
            generate_images_with_masks( args, pre_early, generator, args.epoch, \
                                        write_log = True, \
                                        output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch), \
                                        save_file = output_dir + "mk_stats.csv")
        else:
            generate_masks_with_stats ( args, pre_early, generator, args.epoch, \
                                        write_log = True, \
                                        output_dir = "{0}/generated_images/ep_{1}/".format(args.result_dir, args.epoch), \
                                        save_file = output_dir + "mk_stats.csv")
        
        print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")
        if not args.time_intensities and not args.random_patches_metrics: 
            subj = "✅   Masks metrics Completed! \n"
            notify(notifier, args.exp_name, subj, body, prev_time)
            exit()
    

    exit()



def main(): 
    #
    parser = argparse.ArgumentParser(description= "Training GANs using CA loss")
    
    # Configs  
    parser.add_argument("--exp_name", type=str, default="exp_1", help="name of the experiment")
    parser.add_argument("--gpus", type = str, default = None, help="GPUs to use")
    parser.add_argument("--result_dir", type = str, default = "Results/", help = "Results path. Default = %(default)s")
    parser.add_argument("--images_dir", type = str, default = "Results/", help = "Path containing generated images. Default = %(default)s")
    parser.add_argument("--pixel_metrics", help="Pixel metrics computation mode (default: False)", default=None, action="store_true")
    parser.add_argument("--time_intensities", help="Time intensity computation mode (default: False)", default=None, action="store_true")
    parser.add_argument("--random_patches_metrics", help="Random patching computation mode (default: False)", default=None, action="store_true")
    parser.add_argument("--random_patches_points", help="Random point generator for patches metrics (default: False)", default=None, action="store_true")
    parser.add_argument("--mask_metrics", help="Pixel metrics computation mode (default: False)", default=None, action="store_true")
    parser.add_argument("--lpips_net", help="Model to compute LPIPS metrics (default: %(default)s)", type=str, default="alex", choices=['alex', 'vgg'])
    parser.add_argument("--mode", help="Mode to compute metrics (default: %(default)s)", type=str, default="from_model", choices=['from_model', 'from_path'])

    # Custom configs 
    parser.add_argument("--model", help="Model to use.", default = None, choices=["UNet", "Pix2Pix", "ResViT", "PAN", "TSGAN", "EaGAN", "DualPix2Pix"])
    parser.add_argument("--notification_mode", help="Setup mean for notification", default = 'Telegram', choices=["Gmail", "Telegram"])
    parser.add_argument("--no_notifications", help="Point extraction mode (default: False)", default=False, action="store_true")

    # Params for random patches 
    parser.add_argument("--random_roi_size", type=int, default = 64, help = "Random patch size") 
    parser.add_argument("--random_patch_seed", type=int, default = 0, help = "Random seed for patch generation") 
    parser.add_argument("--random_patches_per_image", type=int, default = 1, help = "Num of random patches per image")
    parser.add_argument("--random_points_patches", type=str, default = None, help = "csv file with the points to extract patches and compute metrics")

    # Dataset params
    parser.add_argument("--image_size", type=int, help = "Input image size")
    parser.add_argument("--roi_size", type=int, default = 64, help = "Input ROI size for DataLoader")
    parser.add_argument("--input", help = "Input sequence, space separated", nargs='+') #type=list
    parser.add_argument("--output", help = "Output sequence ")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--dataset_name", type=str, default="dce-mri", help="Dataset name")
    parser.add_argument("--data_dir", type = str, default = "Data/", help="Data dir path")
    parser.add_argument("--quality", default = None, help = "Quality of images (duke data only)")
    parser.add_argument("--normalization", help="Type of normalization", default = None, choices=["z_score", "min_max", "ti_norm", "none"])
    
    # Training params
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation") 
    parser.add_argument("--sample_size", type=int, default=10, help="interval between sampling of images from generators")

    # Initial configs
    args = parser.parse_args()
    if args.gpus: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    setup_configs(args, training=False)
    run_validation(args)


if __name__ == "__main__":
    #
    """
    param = sys.argv.append

    args = "--no_notifications \
            --gpus 0 \
            --epoch 200 \
            --sample_size -1 \
            --dataset_name duke \
            --input pre post_1 --output post_3 --quality 1T \
            --result_dir Exp_base_/ \
            --exp_name Pix_exp_3/ \
            --data_dir Data/Duke/Duke_tiff/ \
            --image_size 256 --channels 1 \
            --num_workers 8 \
            --normalization ti_norm \
            --model Pix2Pix --mask_metrics" #pixel_metrics time_intensities random_patches_metrics mask_metrics
            # --random_roi_size 10 --random_patch_seed 0 --random_patches_per_image 3 \
            # --random_patches_metrics --random_points_patches Results/D1T/points_patches/meta_chkpt_1T.csv  \
    
    for arg in args.split(" "): 
        if arg: param(arg)
    """
    main()

