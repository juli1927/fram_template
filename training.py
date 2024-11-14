
import os
import sys
import time
from colorama import Fore, Back
import random

from utils import *
from dataloader import *

from Models.Pix2Pix import GeneratorUNet, PatchGAN_Discriminator, weights_init_normal
from Models.Mask_Pix2Pix import Generator_Mask_UNet, Mask_PatchGAN_Discriminator
from Models.Mask_R_Pix2Pix import Generator_Mask_R_UNet, Mask_R_PatchGAN_Discriminator
from Models.Mask_R_Pix2Pix_bloque import Generator_Mask_R_UNet_bloque, Mask_R_PatchGAN_Discriminator_bloque
import torchvision.transforms.functional as TF

def run_model(args): 
    #
    import torch
    import torch.nn.functional as F
    
    from torch.autograd import Variable
    import torchvision.transforms as transforms

    print ("GPU is available: {0}".format(torch.cuda.is_available()))

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if args.exps_to_compare:
        args.exps_to_compare.extend([args.result_dir])
    
    if args.model == "UNet":
        # Initialize generator and discriminator
        generator = GeneratorUNet(n_channels = args.channels)
        
        discriminator = None
        # Loss functions
        cuda_models  = [generator]
        cuda_losses = []
    
    elif args.model == "Pix2Pix": # GAN
        # Initialize generator and discriminator
        generator = GeneratorUNet(n_channels = args.channels)
        discriminator = PatchGAN_Discriminator(n_channels = args.channels)
        # Calculate output of image discriminator (PatchGAN)
        patch = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

        # Loss functions
        GAN_loss = torch.nn.MSELoss()
        cuda_models  = [generator, discriminator]
        cuda_losses  = [GAN_loss]
    
    elif args.model == "Mask_UNet": 
        # Initialize generator and discriminator
        generator = Generator_Mask_UNet(n_channels = args.channels)
        
        discriminator = None
        # Loss functions
        cuda_models  = [generator]
        cuda_losses = []
    
    elif args.model == "Mask_Pix2Pix":
        # Initialize generator and discriminator
        generator = Generator_Mask_UNet(n_channels = args.channels)
        discriminator = Mask_PatchGAN_Discriminator(n_channels = args.channels)
        # Calculate output of image discriminator (PatchGAN)
        patch = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

        # Loss functions
        GAN_loss = torch.nn.MSELoss()
        cuda_models  = [generator, discriminator]
        cuda_losses  = [GAN_loss]
        
    elif args.model == "Mask_R_UNet": 
        # Initialize generator and discriminator
        generator = Generator_Mask_R_UNet(n_channels = args.channels)
        
        discriminator = None
        # Loss functions
        cuda_models  = [generator]
        cuda_losses = []
    
    elif args.model == "Mask_R_Pix2Pix":
        # Initialize generator and discriminator
        generator = Generator_Mask_R_UNet(n_channels = args.channels)
        discriminator = Mask_R_PatchGAN_Discriminator(n_channels = args.channels)
        # Calculate output of image discriminator (PatchGAN)
        patch = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

        # Loss functions
        GAN_loss = torch.nn.MSELoss()
        cuda_models  = [generator, discriminator]
        cuda_losses  = [GAN_loss]

    elif args.model == "Mask_R_UNet_bloque": 
        # Initialize generator and discriminator
        generator = Generator_Mask_R_UNet_bloque(n_channels = args.channels)
        
        discriminator = None
        # Loss functions
        cuda_models  = [generator]
        cuda_losses = []
    
    elif args.model == "Mask_R_Pix2Pix_bloque":
        # Initialize generator and discriminator
        generator = Generator_Mask_R_UNet_bloque(n_channels = args.channels)
        discriminator = Mask_R_PatchGAN_Discriminator_bloque(n_channels = args.channels)
        # Calculate output of image discriminator (PatchGAN)
        patch = (1, args.image_size // 2 ** 4, args.image_size // 2 ** 4)

        # Loss functions
        GAN_loss = torch.nn.MSELoss()
        cuda_models  = [generator, discriminator]
        cuda_losses  = [GAN_loss]
    
    else: 
        # Initialize generator and discriminator
        generator = GeneratorUNet(n_channels = args.channels)
        discriminator = None
        cuda_models = [generator]
        cuda_losses = []
    
    # Setup losses
    pixelwise_loss = torch.nn.L1Loss() #args.lambda_pixel = 1 
    cuda_losses.append (pixelwise_loss)

    #novel_loss = torch.nn.L1Loss() #args.lambda_pixel = 1 
    #cuda_losses.append (novel_loss)
    mae = torch.nn.L1Loss() 

    # Move everything to gpu [int(gpu) for gpu in args.gpus.split(",")] len(args.gpus.split(","))
    if args.cuda:
        for model_ in cuda_models: model_.cuda()
        for loss_  in cuda_losses: loss_.cuda()
    
    # Setup notifier and logger
    monitor = Monitor(logs_path = args.result_dir + "/Logs")

    # Load model in case of training. Otherwise, random init
    if   args.restart_from != 0:
        generator.load_state_dict(torch.load("{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, args.restart_from)))
        print (Fore.GREEN + "Weights from checkpoint: {0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, args.restart_from) + Fore.RESET)
        if discriminator != None: 
            discriminator.load_state_dict(torch.load("{0}/saved_models/D_chkp_{1:03d}.pth".format(args.result_dir, args.restart_from)))
            print (Fore.GREEN + "Weights from checkpoint: {0}/saved_models/D_chkp_{1:03d}.pth".format(args.result_dir, args.restart_from) + Fore.RESET)

    elif args.restart_from == 0:
        # Initialize weights 
        if os.path.isfile("{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, args.restart_from)):
            print(Fore.RED + "Model already exist. No retraining!" +  Fore.RESET)
            exit()
        
        generator.apply(weights_init_normal)
        if discriminator != None: 
            discriminator.apply(weights_init_normal)
    
    
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
    
    if args.use_augmentations: 
        augmentation = transforms.ColorJitter(brightness=(0.01,0.03), contrast=0,   saturation=0, hue=0)
        flips = [transforms.RandomHorizontalFlip(p=1),
                 transforms.RandomVerticalFlip(p=1)]
    else:
        augmentations = [None]
    
    # Initialize data loader
    pre_late = Loader (input_sequence = args.input_sequence, output_sequence = args.output_sequence,output_labels = args.output_labels, data_path = args.data_path, 
                             batch_size = args.batch_size, 
                             img_res=(args.image_size, args.image_size),  
                             n_channels = args.channels,
                             train_transforms = transforms_, val_transforms = transforms_, 
                             dataset_name = args.dataset_name, workers=args.num_workers,
                             norm=args.normalization) #poner el dataloader, (punto)
    
    # Initialize data loader
    # pre_early = Loader (input_sequence = args.input[0], output_sequence = args.output, data_path = args.data_dir, quality = args.quality,
    #                          batch_size = args.batch_size, 
    #                          img_res=(args.image_size, args.image_size), roi_size = (args.roi_size,args.roi_size), 
    #                          n_channels = args.channels,
    #                          train_transforms = transforms_, val_transforms = transforms_, roi_transforms = roi_transforms_, 
    #                          dataset_name = args.dataset_name, workers=args.num_workers,
    #                          norm=args.normalization)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    if discriminator != None: 
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Tensor type
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    # Setting up name of logs
    avg_names = ["Avg_Loss/D", "Avg_Loss/G", "Avg_Loss/Adv", 
                 "Avg_Loss/PixelLoss", "Avg_Metric/MAE"]
    
    # ----------
    #  Training
    # ----------
    # hasta aqui debo llegar
    print ("\n" + Fore.BLUE + "[*] -> Starting training...." + Fore.RESET + "\n\n")

    # subj = "🤖   Starting training....\n" 
    # body = "     - Model: {0} \n".format(args.model) + \
    #        "     - Exp name: {0} \n".format(args.exp_name) + \
    #        "     - Normalization: {0} \n".format(args.normalization)
    
    for epoch in range(args.restart_from, args.num_epochs):
        #
        epoch_stats = {}
        for name in avg_names:  epoch_stats[name] = []

        ## Lambda schedulers
        if epoch == args.lcm_epoch_increase:
            args.lambda_cm *= args.lcm_proportion 
        
        for i, batch_pl in enumerate(pre_late.train_loader):
            #
            if args.reduced_training and i*pre_late.batch_size > 1000: break

            # for augmentation in augmentations: 
            #
            real_lab= Variable(batch_pl["label"].type(Tensor))
            real_in  = Variable(batch_pl["in"].type(Tensor))
            real_out = Variable(batch_pl["out"].type(Tensor))

            # if args.quality == '3T' and args.use_augmentations:
            #     flip = np.random.choice(flips)
            #     if np.random.rand() > 0.5: 
            #         real_lab = flip(real_lab)
            #         real_in  = flip(real_in)
            #         real_out = flip(real_out)
            
            # if args.quality == '1T' and args.use_augmentations:
            #     if np.random.rand() > 0.5: 
            #         real_in  = augmentation(real_in)
            
            # if args.use_random_crops:
            #     #
            #     # Random crop
            #     for bidx in range (len(real_ref)):
            #         f, c, h, w = transforms.RandomCrop.get_params(real_ref[bidx], output_size=(int(args.image_size * args.cropping_ratio), int(args.image_size * args.cropping_ratio)))
            #         real_ref[bidx] = transforms.Resize((args.image_size, args.image_size))(TF.crop(real_ref[bidx], f, c, h, w))
            #         real_in [bidx] = transforms.Resize((args.image_size, args.image_size))(TF.crop(real_in [bidx], f, c, h, w))
            #         real_out[bidx] = transforms.Resize((args.image_size, args.image_size))(TF.crop(real_out[bidx], f, c, h, w))


            # ------------------------------------
            #           Train Generator
            # ------------------------------------
            #
            generator.train()

            # # GAN loss 
            if args.model != "Mask_Pix2Pix" and args.model != "Mask_UNet" and args.model != "Mask_R_Pix2Pix" and args.model != "Mask_R_UNet" and args.model != "Mask_R_Pix2Pix_bloque" and args.model != "Mask_R_UNet_bloque":
                fake_out = generator(real_in)
            else: 
                fake_out = generator(real_in, real_lab)
        
            # Adversarial ground truths
            if discriminator is not None:
                valid = Variable(Tensor(np.ones ((real_in.size(0), *patch))), requires_grad=False)
                fake  = Variable(Tensor(np.zeros((real_in.size(0), *patch))), requires_grad=False)

                if args.model != "Mask_Pix2Pix" and args.model != "Mask_UNet" and args.model != "Mask_R_Pix2Pix" and args.model != "Mask_R_UNet" and args.model != "Mask_R_Pix2Pix_bloque" and args.model != "Mask_R_UNet_bloque":
                    fake_pred = discriminator(fake_out, real_in)
                else: 
                    fake_pred = discriminator(fake_out, torch.cat([real_in, real_lab], dim=1))

                loss_GAN = GAN_loss(fake_pred, valid)


            else: loss_GAN = 0

            ## Pixel-wise loss
            loss_pixel = args.lambda_pixel * pixelwise_loss(fake_out, real_out) 
            mae_val = mae(fake_out, real_out)

            ### Training cycle
            optimizer_G.zero_grad()
            # Pixel-wise loss
            loss_G = loss_pixel
            # GAN loss
            loss_G += loss_GAN 

            loss_G.backward()
            optimizer_G.step()
            
            # ------------------------------------
            #          Train Discriminator
            # ------------------------------------

            if discriminator is not None:
                optimizer_D.zero_grad()
                if args.model != "Mask_Pix2Pix" and args.model != "Mask_UNet" and args.model != "Mask_R_Pix2Pix" and args.model != "Mask_R_UNet":
                    # Real loss
                    real_pred = discriminator(real_out, real_in)
                    loss_real = GAN_loss(real_pred, valid)

                    # Fake loss
                    fake_pred = discriminator(fake_out.detach(), real_in)
                    loss_fake = GAN_loss(fake_pred, fake)

                else: 
                    # Real loss
                    real_pred = discriminator(real_out, torch.cat([real_in, real_lab], dim=1))
                    loss_real = GAN_loss(real_pred, valid)

                    # Fake loss
                    fake_pred = discriminator(fake_out.detach(), torch.cat([real_in, real_lab], dim=1))
                    loss_fake = GAN_loss(fake_pred, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)
                loss_D.backward()
                optimizer_D.step()

            else : loss_D = 0

            # ------------------------------------
            #             Log Progress
            # ------------------------------------
            if discriminator is not None: 
                losses = [loss_GAN.item(), loss_D.item(), loss_G.item(), loss_pixel.item()]
                # if args.model == 'TSGAN': losses.append(loss_LD.item())
            else: 
                losses = [loss_G.item(), loss_pixel.item()]
            
            if i % args.log_epoch == 0:
                # Determine approximate time left

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Ad: %f, D: %f, G: %f, Px: %f, MAE: %f]" % (
                        epoch, args.num_epochs,                 # [Epoch %d/%d]
                        i*pre_late.batch_size, len(pre_late),   # [Batch %d/%d]
                        loss_GAN.item() if loss_GAN else 0,     # Ad: %f,
                        loss_D.item() if loss_D else 0,         # D: %f
                        loss_G.item(),                          # G: %f, 
                        loss_pixel.item(),                      # Px: %f,
                        mae_val.item()                          # MAE: %f
                      )
                )

                # save batch stats to logger
                tb_logs = [
                            loss_D.item() if loss_D else 0,
                            loss_G.item(),
                            loss_GAN.item() if loss_GAN else 0,
                            loss_pixel.item(), 
                            mae_val.item()
                          ]
      
                for name, value in zip(avg_names, tb_logs): 
                    epoch_stats[name].append(value)
                
                # monitor.write_logs(tb_names, tb_logs, (epoch*args.batch_size)+i)
        
        # save avg stats to logger
        avg_logs = []
        for name in avg_names: avg_logs.append(np.mean(epoch_stats[name]))
        monitor.write_logs(avg_names, avg_logs, epoch)

        
        # Shuffle train data everything
        # data_loader.on_epoch_end(shuffle = "train")
        
        # Save model checkpoints
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            #
            os.makedirs("%s/saved_models/" % (args.result_dir), exist_ok = True)
            torch.save(generator.state_dict(), "{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, epoch))
            if discriminator != None: 
                torch.save(discriminator.state_dict(), "{0}/saved_models/D_chkp_{1:03d}.pth".format(args.result_dir, epoch))

            # subj =  "🚩   Checkpoint created...\n" 
            # body =  "     - Model: {0} \n".format(args.model) + \
            #         "     - Exp name: {0} \n".format(args.exp_name) + \
            #         "     - Epoch: {0} \n".format(epoch)
            # notify(notifier, args.exp_name, subj, body, prev_time)

        # If at sample interval save image
        if epoch % args.sample_interval == 0:
            sample_images(args, pre_late, generator, epoch, shuffled = False, logger = monitor)
            # subj =  "🚀   Sampling images..\n" 
            # body =  "     - Model: {0} \n".format(args.model) + \
            #         "     - Exp name: {0} \n".format(args.exp_name) + \
            #         "     - Epoch: {0} \n".format(epoch)
            # notify(notifier, args.exp_name, subj, body, prev_time)
            
            # if args.exps_to_compare:
            #     #args.exps_to_compare.extend([args.result_dir])
            #     plots = monitor.plot_logs(list_dirs = args.exps_to_compare)
            #     notifier.notify_plot(plot = plots, names = args.exps_to_compare)


    # subj = "✅   Training completed! \n" 
    # body = "     - Model: {0} \n".format(args.model) + \
    #        "     - Exp name: {0} \n".format(args.exp_name) + \
    #        "     - Normalization: {0} \n".format(args.normalization)
    
    # notify(notifier, args.exp_name, subj, body, prev_time)

    # if args.exps_to_compare:
    #     # args.exps_to_compare.extend([args.result_dir])
    #     plots = monitor.plot_logs(list_dirs = args.exps_to_compare)
    #     notifier.notify_plot(plot = plots, names = args.exps_to_compare)
    
    print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")




