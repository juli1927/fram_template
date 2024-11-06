
import os
import sys
import time
from colorama import Fore, Back
import random

from utils import *
from dataloader import *

from Models.Pix2Pix import GeneratorUNet, PatchGAN_Discriminator, weights_init_normal
from skimage import filters
import torchvision.transforms.functional as TF

def run_model(args): 
    #
    import torch
    import torch.nn.functional as F
    
    from torch.autograd import Variable
    import torchvision.transforms as transforms

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # if args.exps_to_compare:
    args.exps_to_compare.extend([args.result_dir])
    
    if args.model == "UNet":
        # Initialize generator and discriminator
        generator = GeneratorUNet(n_channels = args.channels)
        discriminator = None
        # Loss functions
        cuda_models  = [generator]
        cuda_losses = []
    
    elif args.model == "Pix2Pix":
        # Initialize generator and discriminator
        generator = GeneratorUNet(n_channels = args.channels)
        discriminator = PatchGAN_Discriminator(n_channels = args.channels)
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

    novel_loss = torch.nn.L1Loss() #args.lambda_pixel = 1 
    cuda_losses.append (novel_loss)
    mae = torch.nn.L1Loss() 

    # Move everything to gpu [int(gpu) for gpu in args.gpus.split(",")] len(args.gpus.split(","))
    if args.cuda:
        for model_ in cuda_models: model_.cuda()
        for loss_  in cuda_losses: loss_.cuda()
    
    # Setup notifier and logger
    monitor = Monitor(logs_path = args.result_dir + "/Logs")

    # Load model in case of training. Otherwise, random init
    if   args.restart_from != 0:
        if args.model != "ResViT":
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
    pre_late = Loader (input_sequence = args.input[0], output_sequence = args.input[1], data_path = args.data_dir, quality = args.quality,
                             batch_size = args.batch_size, 
                             img_res=(args.image_size, args.image_size), roi_size = (args.roi_size,args.roi_size), 
                             n_channels = args.channels,
                             train_transforms = transforms_, val_transforms = transforms_, roi_transforms = roi_transforms_, 
                             dataset_name = args.dataset_name, workers=args.num_workers,
                             norm=args.normalization)
    
    # Initialize data loader
    pre_early = Loader (input_sequence = args.input[0], output_sequence = args.output, data_path = args.data_dir, quality = args.quality,
                             batch_size = args.batch_size, 
                             img_res=(args.image_size, args.image_size), roi_size = (args.roi_size,args.roi_size), 
                             n_channels = args.channels,
                             train_transforms = transforms_, val_transforms = transforms_, roi_transforms = roi_transforms_, 
                             dataset_name = args.dataset_name, workers=args.num_workers,
                             norm=args.normalization)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    if discriminator != None: 
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Tensor type
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

    # Setting up name of logs
    avg_names = ["Avg_Loss/D", "Avg_Loss/G", "Avg_Loss/Adv", 
                 "Avg_Loss/PixelLoss", "Avg_Metric/MAE", "Avg_Metric/Enh-Diff"]
    
    # Setting up other of logs
    cmp_names = ["cmp_Loss/LD", "cmp_Loss/AddLosses", "cmp_Loss/CannyLoss", 
                 "cmp_Loss/CannyLoss-ROI", "cmp_Loss/PixelLoss-ROI",
                 "cmp_Loss/FeatureLoss", "cmp_Loss/FeatureLoss-ROI"]
    
    # ----------
    #  Training
    # ----------

    print ("\n" + Fore.BLUE + "[*] -> Starting training...." + Fore.RESET + "\n\n")

    subj = "🤖   Starting training....\n" 
    body = "     - Model: {0} \n".format(args.model) + \
           "     - Exp name: {0} \n".format(args.exp_name) + \
           "     - Normalization: {0} \n".format(args.normalization)
    
    prev_time = time.time()

    for epoch in range(args.restart_from, args.num_epochs):
        #
        epoch_stats = {}
        for name in avg_names:  epoch_stats[name] = []

        ## Lambda schedulers
        if epoch == args.lcm_epoch_increase:
            args.lambda_cm *= args.lcm_proportion 
        
        for i, (batch_pl, batch_pe) in enumerate(zip(pre_late.train_loader, pre_early.train_loader)):
            #
            if args.reduced_training and i*pre_late.batch_size > 1000: break

            # for augmentation in augmentations: 
            #
            real_ref = Variable(batch_pl["in"].type(Tensor))
            real_in  = Variable(batch_pe["in"].type(Tensor))
            real_out = Variable(batch_pe["out"].type(Tensor))

            if args.quality == '3T' and args.use_augmentations:
                flip = np.random.choice(flips)
                if np.random.rand() > 0.5: 
                    real_ref = flip(real_ref)
                    real_in  = flip(real_in)
                    real_out = flip(real_out)
            
            if args.quality == '1T' and args.use_augmentations:
                if np.random.rand() > 0.5: 
                    real_in  = augmentation(real_in)
            
            if args.use_random_crops:
                #
                # Random crop
                for bidx in range (len(real_ref)):
                    f, c, h, w = transforms.RandomCrop.get_params(real_ref[bidx], output_size=(int(args.image_size * args.cropping_ratio), int(args.image_size * args.cropping_ratio)))
                    real_ref[bidx] = transforms.Resize((args.image_size, args.image_size))(TF.crop(real_ref[bidx], f, c, h, w))
                    real_in [bidx] = transforms.Resize((args.image_size, args.image_size))(TF.crop(real_in [bidx], f, c, h, w))
                    real_out[bidx] = transforms.Resize((args.image_size, args.image_size))(TF.crop(real_out[bidx], f, c, h, w))


            # ------------------------------------
            #           Train Generator
            # ------------------------------------

            if args.model != "ResViT":
                #
                generator.train()

                # # GAN loss 
                # fake_out = generator(real_in)
                
                if args.enhancement_maps: 
                    #
                    if np.random.rand() > 0.5: 
                        msk = (real_in > 0.1*real_in.max())*1.0
                        r_sub = (real_out - real_in)*msk
                        r_enh = ((r_sub)/((real_in*msk)))*100
                        r_enh[torch.isnan(r_enh)] = 0
                        r_dec = (r_enh < 0) * 1.0
                        r_inc = (r_enh > 0) * 1.0
                        m_enh = (r_inc + r_dec) * real_out
                        m_enh = m_enh.repeat(1, 3, 1, 1)

                        emaps = e_net(m_enh) 
                        fake_out = generator(real_in, emaps)
                    else: 
                        fake_out = generator(real_in)
                else: 
                    fake_out = generator(real_in)

                
                # Adversarial ground truths
                if discriminator is not None:
                    valid = Variable(Tensor(np.ones ((real_in.size(0), *patch))), requires_grad=False)
                    fake  = Variable(Tensor(np.zeros((real_in.size(0), *patch))), requires_grad=False)

                    if args.model == "TSGAN":
                        local_valid = Variable(Tensor(np.ones ((real_in.size(0), *local_patch))), requires_grad=False)
                        local_fake  = Variable(Tensor(np.zeros((real_in.size(0), *local_patch))), requires_grad=False)
                        roi_fake = []

                        for z in range (len(fake_out)): 
                            pt = [batch_pe["pt_roi_out"][0][0][z], batch_pe["pt_roi_out"][0][1][z]], [batch_pe["pt_roi_out"][1][0][z], batch_pe["pt_roi_out"][1][1][z]]
                            roi_fake += [extract_roi_from_points(fake_out[z].cpu().detach().numpy(), pt)]
                        
                        roi_fake = Variable(torch.from_numpy(np.array(roi_fake)).type(Tensor))

                        canny_fake_roi = canny.apply(roi_fake)
                        canny_reali_roi = canny.apply(Variable(batch_pe["roi_in"].type(Tensor)))
                        canny_realo_roi = canny.apply(Variable(batch_pe["roi_out"].type(Tensor)))

                        roi_fake_c = torch.cat([roi_fake, canny_fake_roi], axis=1)
                        roi_reali_c = torch.cat([Variable(batch_pe["roi_in"].type(Tensor)), canny_reali_roi], axis=1)
                        roi_realo_c = torch.cat([Variable(batch_pe["roi_out"].type(Tensor)), canny_realo_roi], axis=1)
                        
                        local_fake_pred, local_fake_feat = local_discriminator(roi_fake_c, roi_reali_c, feature_matching=True)
                        _, local_real_feat = local_discriminator(roi_realo_c, roi_reali_c, feature_matching=True)
                        loss_local = local_loss(local_fake_pred, local_valid) 

                        #####
                        canny_fake = canny.apply(fake_out)
                        canny_reali = canny.apply(real_in)
                        canny_realo = canny.apply(real_out)
                        fake_out = torch.cat([fake_out, canny_fake], axis=1)
                        real_in = torch.cat([real_in, canny_reali], axis=1)
                        real_out = torch.cat([real_out, canny_realo], axis=1)

                        canny_fake = canny_fake.type(Tensor)
                        canny_reali = canny_reali.type(Tensor)
                        canny_realo = canny_realo.type(Tensor)
                        canny_fake_roi = canny_fake_roi.type(Tensor)
                        canny_reali_roi = canny_reali_roi.type(Tensor)
                        canny_realo_roi = canny_realo_roi.type(Tensor)

                        loss_canny = args.lambda_canny * canny_loss(canny_fake, canny_realo) ## Edge Im
                        loss_canny_roi = args.lambda_canny * canny_loss_roi(canny_fake_roi, canny_realo_roi) # Edge ROI
                        loss_roi = args.lambda_pixel * roi_loss(roi_fake, Variable(batch_pe["roi_out"].type(Tensor)))
                        loss_roi_features = args.lambda_features * feature_loss(local_fake_feat.mean(), local_real_feat.mean())

                        add_losses = loss_canny + loss_canny_roi + loss_local + loss_roi + loss_roi_features
                    
                    #if args.model == "TSGAN":
                        fake_pred, fake_feat = discriminator(fake_out, real_in, feature_matching=True)
                        _, real_feat = discriminator(real_out, real_in, feature_matching=True)
                        loss_features = args.lambda_features * feature_loss(fake_feat.mean(), real_feat.mean())
                        add_losses += loss_features
                    else: 
                        fake_pred = discriminator(fake_out, real_in)
                    
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

                if args.model == 'TSGAN': loss_G += add_losses ########-------->

                # msk = (real_in > 0.1*real_in.max())*1.0
                # r_sub = (real_out - real_in)*msk
                # r_enh = ((r_sub)/((real_in*msk)))*100
                # r_enh[torch.isnan(r_enh)] = 0

                # f_sub = (fake_out - real_in)*msk
                # f_enh = ((f_sub)/((real_in*msk)))*100
                # f_enh[torch.isnan(f_enh)] = 0

                # r_dec = (r_enh < 0) * 1.0
                # r_inc = (r_enh > 0) * 1.0

                # f_dec = (f_enh < 0) * 1.0
                # f_inc = (f_enh > 0) * 1.0

                # loss_novel  = args.lambda_novel * novel_loss(f_dec, r_dec) 
                # loss_novel += args.lambda_novel * novel_loss(f_inc, r_inc) 

                if args.novel_loss: 
                    loss_G += loss_novel 
                else:
                    loss_novel = 0
                
                loss_G.backward()
                optimizer_G.step()
                
                # ------------------------------------
                #          Train Discriminator
                # ------------------------------------

                if discriminator is not None:
                    optimizer_D.zero_grad()
                    # Real loss
                    real_pred = discriminator(real_out, real_in)
                    loss_real = GAN_loss(real_pred, valid)

                    # Fake loss
                    fake_pred = discriminator(fake_out.detach(), real_in)
                    loss_fake = GAN_loss(fake_pred, fake)

                    # Total loss
                    loss_D = 0.5 * (loss_real + loss_fake)
                    loss_D.backward()
                    optimizer_D.step()

                    if args.model == 'TSGAN':
                        optimizer_LD.zero_grad()
                        # Real loss
                        real_pred = local_discriminator(roi_realo_c, roi_reali_c) 
                        loss_real_ld = local_loss(real_pred, local_valid)

                        # Fake loss
                        fake_pred = local_discriminator(roi_fake_c, roi_reali_c)
                        loss_fake_ld = local_loss(fake_pred, local_fake)

                        # Total loss
                        loss_LD = 0.5 * (loss_real_ld + loss_fake_ld)
                        loss_LD.backward()
                        optimizer_LD.step()

                else : loss_D = 0

            else: 
                # 
                resvit_model.set_input(real_in, real_out)
                resvit_model.optimize_parameters()
                loss_GAN, loss_G, loss_D, loss_pixel, mae_val = resvit_model.get_current_errors()
                
            # ------------------------------------
            #             Log Progress
            # ------------------------------------
            if discriminator is not None: 
                losses = [loss_GAN.item(), loss_D.item(), loss_G.item(), loss_pixel.item()]
                # if args.model == 'TSGAN': losses.append(loss_LD.item())
            else: 
                losses = [loss_G.item(), loss_pixel.item()]
            
            for any_loss in losses:
                if np.isnan(any_loss):
                    print ("\n" + Fore.RED + "[X] -> Stop training! NaN Gradients" + Fore.RESET + "\n\n")

                    subj = "❌   Error! Stop training \nNaN Gradients\n" 
                    body = "     - Model: {0} \n".format(args.model) + \
                           "     - Exp name: {0} \n".format(args.exp_name) + \
                           "     - Normalization: {0} \n".format(args.normalization)
                    notify(notifier, args.exp_name, subj, body, prev_time)
                    exit()
            
            if i % args.log_epoch == 0:
                # Determine approximate time left
                elapsed_time = time.time() - prev_time
                hours = elapsed_time // 3600; elapsed_time = elapsed_time - 3600 * hours
                minutes = elapsed_time // 60; seconds = elapsed_time - 60 * minutes

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [Ad: %f, D: %f, G: %f, Px: %f, MAE: %f, EnhD: %f]" % (
                        epoch, args.num_epochs,                 # [Epoch %d/%d]
                        i*pre_early.batch_size, len(pre_early), # [Batch %d/%d]
                        loss_GAN.item() if loss_GAN else 0,     # Ad: %f,
                        loss_D.item() if loss_D else 0,         # D: %f
                        loss_G.item(),                          # G: %f, 
                        loss_pixel.item(),                      # Px: %f,
                        mae_val.item(),                         # MAE: %f
                        loss_novel.item() if loss_novel else 0  # EnhD: %f
                      )
                )

                # save batch stats to logger
                tb_logs = [
                            loss_D.item() if loss_D else 0,
                            loss_G.item(),
                            loss_GAN.item() if loss_GAN else 0,
                            loss_pixel.item(), 
                            mae_val.item(),
                            loss_novel.item() if loss_novel else 0 
                          ]
                
                if args.model == 'TSGAN':
                    #
                    cm_logs = [
                                loss_LD.item(),
                                add_losses.item(), 
                                loss_canny.item(),
                                loss_canny_roi.item(),
                                loss_roi.item(),
                                loss_features.item(),
                                loss_roi_features.item()
                            ]
                
                for name, value in zip(avg_names, tb_logs): 
                    epoch_stats[name].append(value)
                
                if args.model == 'TSGAN':
                    for name, value in zip(cmp_names, cm_logs): 
                        epoch_stats[name].append(value)
                
                # monitor.write_logs(tb_names, tb_logs, (epoch*args.batch_size)+i)
        
        # save avg stats to logger
        avg_logs = []
        for name in avg_names: avg_logs.append(np.mean(epoch_stats[name]))
        monitor.write_logs(avg_names, avg_logs, epoch)

        if args.model == 'TSGAN':
            avg_logs = []
            for name in cmp_names: avg_logs.append(np.mean(epoch_stats[name]))
            monitor.write_logs(cmp_names, avg_logs, epoch)
        
        # Shuffle train data everything
        # data_loader.on_epoch_end(shuffle = "train")
        
        # Save model checkpoints
        if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
            #
            os.makedirs("%s/saved_models/" % (args.result_dir), exist_ok = True)
            torch.save(generator.state_dict(), "{0}/saved_models/G_chkp_{1:03d}.pth".format(args.result_dir, epoch))
            if discriminator != None: 
                torch.save(discriminator.state_dict(), "{0}/saved_models/D_chkp_{1:03d}.pth".format(args.result_dir, epoch))
                if args.model == 'TSGAN': 
                    torch.save(local_discriminator.state_dict(), "{0}/saved_models/LD_chkp_{1:03d}.pth".format(args.result_dir, epoch))
            subj =  "🚩   Checkpoint created...\n" 
            body =  "     - Model: {0} \n".format(args.model) + \
                    "     - Exp name: {0} \n".format(args.exp_name) + \
                    "     - Epoch: {0} \n".format(epoch)
            notify(notifier, args.exp_name, subj, body, prev_time)

        # If at sample interval save image
        if epoch % args.sample_interval == 0:
            sample_images(args, pre_early, generator, epoch, shuffled = False, logger = monitor)
            subj =  "🚀   Sampling images..\n" 
            body =  "     - Model: {0} \n".format(args.model) + \
                    "     - Exp name: {0} \n".format(args.exp_name) + \
                    "     - Epoch: {0} \n".format(epoch)
            notify(notifier, args.exp_name, subj, body, prev_time)
            
            if args.exps_to_compare:
                #args.exps_to_compare.extend([args.result_dir])
                plots = monitor.plot_logs(list_dirs = args.exps_to_compare)
                notifier.notify_plot(plot = plots, names = args.exps_to_compare)


    subj = "✅   Training completed! \n" 
    body = "     - Model: {0} \n".format(args.model) + \
           "     - Exp name: {0} \n".format(args.exp_name) + \
           "     - Normalization: {0} \n".format(args.normalization)
    
    notify(notifier, args.exp_name, subj, body, prev_time)

    if args.exps_to_compare:
        # args.exps_to_compare.extend([args.result_dir])
        plots = monitor.plot_logs(list_dirs = args.exps_to_compare)
        notifier.notify_plot(plot = plots, names = args.exps_to_compare)
    
    print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")


    ######
    # Run validation
    ######

    #"""
    if args.resvit_mode == "finetune":
        #
        sys.argv = ['file.py']
        param = sys.argv.append
        
        args = "--gpus 0 \
                --epoch 200 \
                --sample_size -1 \
                --dataset_name duke \
                --input pre post_1 --output post_3 --quality 1T \
                --exp_name D1T/E2/RV_/ \
                --data_dir Data/Duke/Duke_tiff/ \
                --image_size 256 --channels 1 \
                --num_workers 8 \
                --model ResViT --normalization referenced \
                --pixel_metrics --time_intensities \
                --random_patches_metrics --random_roi_size 8 \
                --random_points_patches Results/D3T/points_patches/points_8x8.csv " # --time_intensities --pixel_metrics 
                # --random_patches_points \
                # --random_roi_size 8 --random_patch_seed 0 --random_patches_per_image 3

        for arg in args.split(" "): 
            if arg: param(arg)
        #"""
        
        main(generator=generator)



