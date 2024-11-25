 
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

from utils import setup_configs
from training import run_model


def main(): 
    #
    parser = argparse.ArgumentParser(description= "Training GANs using CA loss")
    
    # Configs  
    parser.add_argument("--exp_name", type=str, default="exp_1", help="name of the experiment")
    parser.add_argument("--gpus", type = str, default = "0", help="GPUs to use")
    parser.add_argument("--data_dir", type = str, default = "Data/", help="Data dir path")
    parser.add_argument("--result_dir", type = str, default = "Results/", help = "Results path. Default = %(default)s")

    # Dataset params
    parser.add_argument("--image_size", type=int, help = "Input image size")
    parser.add_argument("--roi_size", type=int, default = 64, help = "Input ROI size for DataLoader")
    parser.add_argument("--input", help = "Input sequence, space separated", nargs='+') #type=list
    parser.add_argument("--output", help = "Output sequence ") # debe ser igual al de input, input, output y labels que sean string
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--dataset_name", type=str, default="dce-mri", help="Dataset name")
    parser.add_argument("--quality", default = None, help = "Quality of images (duke data only)")
    parser.add_argument("--use_augmentations", help="Use augmentations during training (default: %(default)s)", default=False, action="store_true")
    parser.add_argument("--use_random_crops", help="Use random croppings for training (default: %(default)s)", default=False, action="store_true")
    parser.add_argument("--cropping_ratio", help="Ratio for the croppings (default: %(default)s)", type=float, default=0.85)

    parser.add_argument("--normalization", help="Type of normalization", default = 'ti_norm', choices=["z_score", "min_max", "ti_norm", "none"])
    
    parser.add_argument("--input_sequence", type=str,help="Input sequence")
    parser.add_argument("--output_sequence", type=str,  help="Output sequence")
    parser.add_argument("--output_labels", type=str,  help="Output labels")
    parser.add_argument("--data_path", type=str,  help="Data path")   

    #### Training params
    parser.add_argument("--model", help="Model to use.", default = None, choices=["Pix2Pix", "UNet", "Mask_Pix2Pix", "Mask_UNet", "Mask_R_Pix2Pix", "Mask_R_UNet", "Mask_R_Pix2Pix_bloque", "Mask_R_UNet_bloque"])
    parser.add_argument("--restart_from", type=int, default=0, help="Restart training from epoch")
    parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    #parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation") 
    parser.add_argument("--sample_size", type=int, default=10, help="interval between sampling of images from generators")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between sampling of images from generators")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints. Default = %(default)s (no save)")
    parser.add_argument("--resvit_mode", help="For ResViT, define mode (pretrain CNN, finetune ViT)", default = None, choices=["pretrain", "finetune"])

    # Additional losses and lambda schedulers
    parser.add_argument("--lambda_pixel", type=float, default=1.0, help="Weight for the Reconstruction loss") 
    parser.add_argument("--lambda_canny", type=float, default=1.0, help="Weight for the Canny loss")
    parser.add_argument("--lambda_features", type=float, default=1.0, help="Weight for the matching feature loss")
    parser.add_argument("--lcm_proportion", type=float, default=1, help="Proportion increase for lambda CM. Default = %(default)s")
    parser.add_argument("--lcm_epoch_increase", type=float, default=-1, help="Increase lambda CM every epochs. Default = disabled")
    parser.add_argument("--lambda_novel", type=float, default=1.0, help="Weight for the Novel loss") 
    parser.add_argument("--novel_loss", help="Use novel loss or not (default: %(default)s)", default=False, action="store_true")
    parser.add_argument("--enhancement_maps", help="Use enhancement maps or not (default: %(default)s)", default=False, action="store_true")
    
    ##########
    # Additional options 
    parser.add_argument("--reduced_training", help="limit training num of samples (default: %(default)s)", default=False, action="store_true")
    parser.add_argument("--notification_mode", help="Setup mean for notification", default = 'Telegram', choices=["Gmail", "Telegram"])
    parser.add_argument("--no_notifications", help="Don't send notifications (default: %(default)s)", default=False, action="store_true")
    parser.add_argument("--random_seed", type=int, default=1, help="Setup random seed for reproducibility")
    parser.add_argument("--log_epoch", type=int, default=5, help="Show progress after epochs")
    parser.add_argument("--exps_to_compare", help = "Input sequence, space separated", nargs='+') #type=list
    ##########

    # Initial configs
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    setup_configs(args)
    run_model(args)


if __name__ == "__main__":
    #
    """ 
    param = sys.argv.append
    
    args = "--no_notifications \
            --gpus 0 \
            --exp_name Prue_P2X \
            --dataset_name BreaDM \
            --input_sequence VIBRANT_IMG \
            --output_sequence VIBRANT+C3_IMG \
            --output_labels VIBRANT+C3_LABEL \
            --data_path BreaDM/ \
            --image_size 256 --channels 1 \
            --batch_size 5 \
            --normalization min_max \
            --num_workers 4 \
            --model Mask_R_Pix2Pix_bloque \
            --num_epochs 100"

    for arg in args.split(" "):
        if arg: param(arg)
    """ 
    #--resvit_mode finetune --restart_from 50 \
    #--reduced_training
    #--novel_loss --lambda_novel 1.0
    main()
