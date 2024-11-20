
import os
import io
import csv
import time
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision.utils import save_image, _log_api_usage_once
import torchvision.transforms as transforms

import json
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from colorama import Fore
import lpips

from torch.utils.tensorboard import SummaryWriter

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests

## Additional dependencies
from metrics import *

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


##
## Tensorboard logs
## 
class Monitor:
    #
    def __init__(self, logs_path = "Logs/"):
        #
        self.logs_path = logs_path
        self.tb = self.create_logs(logs_path)

    def create_logs(self, log_path = "/Logs"):
        #tensorboard = create_file_writer(log_path)
        tensorboard = SummaryWriter(log_dir = log_path)
        return tensorboard

    def write_logs(self, names, logs, current_batch):
        #
        if self.tb:
            for name, value in zip(names, logs):
                self.tb.add_scalar(name, value, current_batch)
                #self.tb.add_image('images', grid, 0)
            """
            with self.tb.as_default():
                for name, value in zip(names, logs):
                    scalar(name, value, step = current_batch)
                    self.tb.flush()
            """
        else:
            raise NameError
            pass
    
    def write_images(self, images, current_batch, custom_name='sample'):
        #
        if self.tb:
            for i, image in enumerate(images):
                # self.tb.add_scalar(name, value, current_batch)
                self.tb.add_image('{0}_{1}'.format(custom_name, i), np.array(image.convert("RGB")), current_batch, dataformats='HWC')
        else:
            raise NameError
            pass
    
    def plot_logs(self, list_dirs, list_tags=['Avg_Metric/MAE',]):
        ##################################
        # Configs 
        ##################################

        # list_tags = ["Loss/G/loss", "Loss/D/loss", "Loss/G/logistic_ns_loss", "Loss/G/feature_consistency_loss",]
        # list_output_tags  = ["-G_Loss", "-D_Loss", "-G_log_Loss", "-G_FC_Loss",]
        verbose = 1

        ##################################
        dropped = []
        to_plot = []

        for log in list_dirs: #os.listdir(list_dirs): 
            #
            #print (log)
            # printlist = []
            event = os.listdir(log + "/Logs/")
            log = log + "/Logs/" + event[0]
            try: 
                event_accumulator = EventAccumulator(log)
                event_accumulator.Reload()
                
                for tag in list_tags:
                    #
                    #tags = event_accumulator.Tags(); print (tags)
                    events = event_accumulator.Scalars(tag); 
                    x = [x.step for x in events]
                    y = [x.value for x in events]
                    # name = log [log.find("_")+1 : log.rfind("-0")] + output_tag + ".csv"
                    #name = log [: log.rfind("-0")] + output_tag + ".csv"

                    df = pd.DataFrame({"Step": x, "Value": y})
                    to_plot.append(df)
                    # df.to_csv(output_path + name, index=False) 
            except Exception as ex: 
                if   verbose == 0: 
                    dropped.append("--> [!] Dropped: {0}".format(log))
                elif verbose == 1: 
                    dropped.append("--> [!] Dropped: {0} \n\tLog: {1}".format(log, ex))
            # print ("for {0}, saved: {1}".format(log, printlist))
        print()
        print ("------------- ")

        if dropped: 
            for d in dropped: print(d)
        
        return to_plot


##
## Nofitications via email
## 
class Gmail_Notifier ():
    def __init__ (self, sender, key_file, receiver = None):
        self.sender = sender
        self.receiver = sender if receiver == None else receiver 
        self.mode = "Gmail"
        
        with open(key_file) as f:
            for line in f:
                tups=line.strip().split("=")
                if tups[0] == "KEY": self.key=tups[1]
    
    
    def notify(self, subject, content = "Notifier"):
        #
        message = MIMEMultipart()
        
        # Information 
        message['From'] = self.sender
        message['To'] = self.receiver
        message['Subject'] = subject
        #The body and the attachments for the mail
        message.attach(MIMEText(content, 'plain'))
        text = message.as_string()
        
        #Create SMTP session for sending the mail
        session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        session.starttls() #enable security
        session.login(self.sender, self.key) #login with mail_id and password
        session.sendmail(self.sender, self.receiver, text)
        session.quit()
        
        # print("Notification sent!")
        # print ("\n" + Fore.GREEN + "[✓] -> Notification sent!" + Fore.RESET + "\n\n")


##
## Nofitications via Telegram
##
class Telegram_Notifier ():
    def __init__ (self, key_file, send_notifications=True):
        #
        self.send_notifications = send_notifications
        self.mode = "Telegram"
        self.api_url = 'https://api.telegram.org/bot{0}/sendMessage?chat_id={1}&parse_mode=Markdown&text={2}'
        self.pic_url = "https://api.telegram.org/bot{0}/sendPhoto?chat_id={1}"
        with open(key_file) as f:
            for line in f:
                tups=line.strip().split("=")
                if   tups[0] == "TG_KEY": self.bot_key=tups[1]
                elif tups[0] == "CHAT_ID": self.chat_id=tups[1]
    
    def notify(self, content = "Notifier"):
        #
        #send_text = 'https://api.telegram.org/bot' + self.bot_key + '/sendMessage?chat_id=' + self.chat_id + '&parse_mode=Markdown&text=' + content
        if self.send_notifications: 
            send_text = self.api_url.format(self.bot_key, self.chat_id, content)
            response = requests.get(send_text)
            return response.json()
        else: 
            r = {}; r['ok'] = True
            return r
    
    def notify_plot (self, plot, names):
        #
        if self.send_notifications:
            #
            _, ax = plt.subplots(figsize=(6,5))
            for df, name in zip(plot, names):
                #
                ax.plot(df['Step'], df['Value'], label=name)
                plt.tight_layout(), plt.legend()
                # plt.savefig("fig.png")
            
            fig = plt.gcf()
            img_ = fig2img(fig, dpi=200)
            img_bytes = io.BytesIO()
            img_.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            files = {"photo": img_bytes}
            send_pic = self.pic_url.format(self.bot_key, self.chat_id)
            response = requests.post(send_pic, files=files)
            return response.json()
        else: 
            r = {}; r['ok'] = True
            return r


##
## Custom scaling transform
## 
class min_max_scaling (torch.nn.Module):
    """Scales a tensor image within [range.min, range.max]
    This transform does not support PIL Image.
    Given range: ``range[0], range[1]`` this transform will scale each input
    ``torch.*Tensor`` i.e.,
    ``output = (range.max - range.min) * ( (input - input.min ) / (tensor.max - tensor.min) ) - range.min ``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        range (sequence): Sequence of min-max values to scale.
        inplace(bool,optional): Bool to make this operation in-place.
    """
    
    def __init__(self, in_range = None, out_range = [-1,1], inplace=False):
        """"""
        super().__init__()
        self.inplace = inplace
        self.out_range = np.asarray(out_range)
        self.in_range  = np.asarray(in_range) if in_range else None
    
    def forward(self, tensor: torch.Tensor, *kwargs) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be scaled.
        Returns:
            Tensor: Scaled Tensor image.
        """
        # in_range = self.in_range if isinstance(self.in_range, np.ndarray) else np.asarray([tensor.min(), tensor.max()]) 
        in_range = self.in_range if isinstance(self.in_range, np.ndarray) else np.asarray([tensor.min().detach().cpu().numpy(), tensor.max().detach().cpu().numpy()]) 
        #return F.normalize(tensor, self.mean, self.std, self.inplace)
        return (self.out_range.max()-self.out_range.min())*(tensor-in_range.min()) / (in_range.max()-in_range.min()) + self.out_range.min()
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(:\n\t(in_range min={self.in_range.min()}, in_range max={self.in_range.max()}), \
                                            \n\t(out_range min={self.out_range.min()}, out_range max={self.out_range.max()}))"

##
## Custom standarization transform
## 
class z_score (torch.nn.Module):
    """Standarize a tensor image given a pair [mean, std].
    It differs from the standard torchvision.Normalize as you can give both
    mean and std during the forward pass.
    This transform does not support PIL Image.
    Given: ``mean, std`` this transform will scale each input
    ``torch.*Tensor`` i.e.,
    ``output = (input - mean)/std``
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor: torch.Tensor, mean = None, std = None) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be standarized.
            mean (float): mean value to s
            andarize.
            std  (float): std value to standarize.
        Returns:
            Tensor: Standarized Tensor image.
        """
        mean = torch.mean(tensor) if mean == None else mean
        std  = torch.std(tensor) if std == None else std 
        return (tensor - mean) / std
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}( \n\t( None )\n\t )"

##
## Custom compose for custom transforms
## 
class Custom_Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.
    
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
        
        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.
    """
    
    def __init__(self, trans):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(self)
        self.transforms = trans
        self.transforms_methods = self.getMethods(transforms)
    
    def __call__(self, img, *kwargs):
        for t in self.transforms:
            if f"{t}"[:f"{t}".find("(")] in self.transforms_methods:
                img = t(img)
            else: 
                img = t(img, *kwargs)
        return img
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
    
    def getMethods(self, clss):
        import types
        result = [ ]
        for var in clss.__dict__:
            val = clss.__dict__[var]
            result.append(var)
        return sorted(result)


##
## Setup all framework
## 
def setup_configs(args, training=True):
    #
    os.makedirs(args.result_dir + "/%s" % args.exp_name, exist_ok=True)
    args.result_dir = args.result_dir + "/%s" % args.exp_name

    #if args.WGAN: args.PatchGAN = False
    if not args.model: 
        raise NotImplementedError ("Which model to use? Implemented: PatchGAN, PA_GAN, PAN")

    if training: 
        save_configs(args)
    
    args.cuda = True if torch.cuda.is_available() and args.gpus else False


##
## Compute elapsed time
## 
def compute_elapsed_time (prev_time):
    elapsed_time = time.time() - prev_time
    hours = elapsed_time // 3600; elapsed_time = elapsed_time - 3600 * hours
    minutes = elapsed_time // 60; seconds = elapsed_time - 60 * minutes
    return hours, minutes, seconds


##
## Setup notifier
## 
def setup_notifier(mode = 'Gmail', send_notifications = True):
    #
    # Initialize email notifier 
    passw = "keys.txt" 
    if mode == 'Gmail':
        sender = 'ruben.fonnegra@pascualbravo.edu.co'
        return Gmail_Notifier (sender, passw)
    else: 
        return Telegram_Notifier (passw, send_notifications)


##
## Send notifications
## 
def notify(notifier, exp_name, subj, body = None, prev_time=None):
    #
    prev_time = time.time() if prev_time==None else prev_time
    hours, minutes, seconds = compute_elapsed_time(prev_time)
    footnote = "Elapsed time: {0:02}:{1:02}:{2:02}".format(int(hours),int(minutes),int(seconds))
    body = footnote if body == None else body + "\n" + footnote
    subj = subj.replace("_", "\_")
    body = body.replace("_", "\_")
    if notifier.mode == "Gmail":
        notifier.notify(subject = subj, content = body)
    elif notifier.mode == "Telegram":
        r = {}; r['ok'] = False; timeout=0
        while r['ok'] == False and timeout<200: 
            r = notifier.notify(content = '*'+subj+'*\n' + body); timeout+=1
        if timeout>=200: 
            notifier.notify(content = '*'+subj+'*\n' + body)


##
## Function to create image grid
## 
def save_images (list_images, output_dir, diffmap = None, image_ax = [0,1,2,4,5,6], diffmap_ax = None, plot_shape = None, figsize=(14, 3), return_image = False): 
    #
    if not plot_shape: 
        num_plots = len(list_images)+1 if diffmap is not None else len(list_images)
        plot_shape = (1,num_plots)
    
    _, axes = plt.subplots(plot_shape[0], plot_shape[1], figsize=figsize) #, figsize = (5,2)
    axes = axes.ravel()

    for i, image in zip(image_ax, list_images):
        # image = np.squeeze(image).transpose(1,2,0) if len(image.shape) >= 3 else np.squeeze(image)
        image = np.squeeze(image)
        image = image.transpose(1,2,0) if len(image.shape) == 3 and (image.shape[0] <= 3) else image
        axes[i].imshow(np.squeeze(image), cmap = "gray") #, vmin=0, vmax=1
        axes[i].set_axis_off(); #print(image.min(), image.max())
    
    if diffmap is not None:
        if diffmap_ax == None: diffmap_ax = -1

        for i, dm in zip(diffmap_ax, diffmap):
            sns.heatmap(np.squeeze(dm), cmap = "hot", ax=axes[i], vmin=0, vmax=1) #, vmin=0, vmax=5
            axes[i].set_axis_off(); axes[i].set_title("Avg diff: {0:0.3f}".format(np.mean(np.squeeze(dm))))

    #plt.margins(0)
    plt.tight_layout(pad=0.1) #
    plt.subplots_adjust(hspace=0.1)
    #plt.savefig(output_dir)
    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save(output_dir)
    plt.close('all')

    if return_image: return fig

##
## Compute time intensity during training (just 2 points)
## 
def time_intensity_training(roi_in, roi_out, roi_fake):
    # Compute the TIC trend to the logs
    m_roi_in  = np.mean(roi_in.cpu().numpy())
    m_roi_out = np.mean(roi_out.cpu().numpy())
    m_roi_fake = np.mean(roi_fake)

    fig, ax = plt.subplots (figsize=(5,5))
    # ax.set_xlim(0, 3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Enhancement")

    ax.plot(range (1,3), [m_roi_in, m_roi_out],  label="Real", linestyle='-',  linewidth = 2)
    ax.plot(range (1,3), [m_roi_in, m_roi_fake], label="Gen ", linestyle='--', linewidth = 2)
    ax.set_title("Time Intensity Curve")
    ax.legend()
    
    plot_as_image = fig2img(fig)
    return plot_as_image

##
## Network params and configuration saving
## 
def save_configs (args):
    #
    with Path("%s/%s" %(args.result_dir, "config.txt")).open("a") as f:
        f.write("\n##############\n   Settings\n##############\n\n")
        args_dict = vars(args)
        for key in args_dict.keys():
            f.write("{0}: {1}, \n".format(json.dumps(key), json.dumps(args_dict[key])))
        f.write("\n")

##
## Image and stats saving in generation mode
## 
def image_lpips(im_real, im_pred, scaler = None, lpips_ = lpips.LPIPS(net='vgg')):
    #
    if scaler: 
        im_real, im_pred = scaler(im_real), scaler(im_pred)

    if im_real.shape[1] == 1:
        im_real = torch.repeat_interleave(im_real, 3, axis=1)
        im_pred = torch.repeat_interleave(im_pred, 3, axis=1)
    
    p_dif = lpips_.forward(im_real, im_pred)

    if isinstance(p_dif, torch.Tensor):
        return float(p_dif.detach().cpu().numpy())
    else:
        return p_dif


##
## Image and stats saving in generation mode
## 
def generate_images_with_stats_deprecated(args, dataloader, generator, epoch, shuffled = True, masked = False, image_level = False, \
                               output_dir = None, write_log = False, exp = "DEF", ca="CA", save_file = "Results/stats.csv"):
        #
        """Saves a generated sample from the validation set"""
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        ce_m = torch.nn.L1Loss()
        difference = True
        
        scaler = min_max_scaling(out_range = [-1,1])
        lpips_ = lpips.LPIPS(net='vgg')
        if args.cuda: lpips_.cuda()

        if args.sample_size == -1:
            args.sample_size = len(dataloader.test_generator)

        if output_dir == None: 
            output_dir = "%s/images/ep%s/" % (args.result_dir, epoch)

        if shuffled: 
            lucky = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
        else: 
            lucky = np.arange(0, args.sample_size)
        
        os.makedirs(output_dir+"imgs/", exist_ok = True)
        os.makedirs(output_dir+"c_fi/", exist_ok = True)
        os.makedirs(output_dir+"c_ro/", exist_ok = True)
        os.makedirs(output_dir+"m_ro/", exist_ok = True)
        m_fi, s_fi, p_fi, m_ro, s_ro, p_ro = [], [], [], [], [], []
        pr_fi, rc_fi, fs_fi, ac_fi, sp_fi  = [], [], [], [], []
        pr_ro, rc_ro, fs_ro, ac_ro, sp_ro  = [], [], [], [], []
        dices, ious = [], []
        ces_fim, ces_roi = [], []
        pdifs_fim, pdifs_roi = [], []
        names = []
        avg_conf_fim, avg_conf_roi = np.zeros([10,10]), np.zeros([10,10])

        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            
            try:
                img = dataloader.test_generator[int(l)]
                real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
                real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
                roi_in  = Variable(img["roi_in" ].type(Tensor)); roi_in = roi_in[None, :]
                roi_out = Variable(img["roi_out"].type(Tensor)); roi_out = roi_out[None, :]

                fake_out = generator(real_in)
                roi_fake = extract_roi_from_points(fake_out.cpu().detach().numpy(), img["pt_roi_out"] )
                
                real_ca = real_in - real_out
                fake_ca = real_in - fake_out
                ce_fim = ce_m(fake_ca, real_ca); ces_fim.append(ce_fim.item())
                
                roi_ca = roi_in - roi_out
                fro_ca = roi_in - torch.tensor(roi_fake, device = roi_in.device)
                ce_roi = ce_m(fro_ca, roi_ca); ces_fim.append(ce_roi.item())

                if difference:
                    diffmap = abs(real_out.data - fake_out.data) 
                    diffroi = abs(roi_out.data.cpu().numpy() - roi_fake[np.newaxis,:,:,:]) 
                    img_sample = [real_in.data.cpu().numpy(), real_out.data.cpu().numpy(), fake_out.data.cpu().numpy(), \
                                roi_in.data.cpu().numpy(), roi_out.data.cpu().numpy(), roi_fake[np.newaxis,:,:,:]]
                    diffmaps = [diffmap.cpu().numpy(), diffroi]
                    
                    ##---- Metrics -----##
                    ##--- FIM ---##
                    m_, s_, p_ = pixel_metrics(real_out.data.cpu().numpy(), fake_out.data.cpu().numpy(), masked=masked)
                    m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)
                    # _, conf_fim, pr_a, rc_a, fsc_a, acc_a, sp_a = estimate_confusion ((real_out.data.cpu().numpy()+1)/2, (fake_out.data.cpu().numpy()+1)/2, \
                    #                                         output_dir + "c_fi/%s.png" % (k), save_figs = True, to_return = "metrics")
                    # pr_fi.append(pr_a), rc_fi.append(rc_a), fs_fi.append(fsc_a), ac_fi.append(acc_a), sp_fi.append(sp_a)
                    # avg_conf_fim = avg_conf_fim + conf_fim.copy()
                    
                    ##--- ROI ---##
                    m_, s_, p_ = pixel_metrics(roi_out.data.cpu().numpy(), roi_fake[np.newaxis,:,:,:])
                    m_ro.append(m_), s_ro.append(s_), p_ro.append(p_)
                    # _, conf_roi, pr_a, rc_a, fsc_a, acc_a, sp_a = estimate_confusion ((roi_out.data.cpu().numpy()+1)/2, (roi_fake[np.newaxis,:,:,:]+1)/2, \
                    #                                         output_dir + "c_ro/%s.png" % (k), save_figs = True, to_return = "metrics")
                    # pr_ro.append(pr_a), rc_ro.append(rc_a), fs_ro.append(fsc_a), ac_ro.append(acc_a), sp_ro.append(sp_a)
                    # avg_conf_roi = avg_conf_roi + conf_roi.copy()
                    
                    # mask_roi_out  = create_mask_otsu(np.squeeze((roi_out.data.cpu().numpy()+1)/2), return_mask = True)
                    # mask_roi_fake = create_mask_otsu(np.squeeze((roi_fake[np.newaxis,:,:,:]+1)/2), return_mask = True)
                    # dice = dice_between_masks (mask_roi_out, mask_roi_fake, k = 1.0)
                    # iou  = IoU_between_masks  (mask_roi_out, mask_roi_fake, k = 1.0)
                    # dices.append(dice), ious.append(iou)
                    # plot_masks(np.squeeze((roi_out.data.cpu().numpy()+1)/2), np.squeeze((roi_fake[np.newaxis,:,:,:]+1)/2), \
                    #                     mask_roi_out, mask_roi_fake, dice, iou, namefile=output_dir + "m_ro/%s.png" % (k))
                    
                    if image_level:
                        roi_fake = torch.from_numpy(roi_fake[np.newaxis,:,:,:])
                        if args.cuda: roi_fake = roi_fake.cuda()

                        pdif_fim = image_lpips(real_out, fake_out, scaler = scaler, lpips_ = lpips_)
                        pdif_roi = image_lpips(roi_out, roi_fake, scaler = scaler, lpips_ = lpips_)
                        pdifs_fim.append(pdif_fim)
                        pdifs_roi.append(pdif_roi)
                    

                    save_images(img_sample, output_dir = output_dir + "imgs/%s.png" % (k), \
                                diffmap = diffmaps, diffmap_ax = [3, 7], plot_shape = (2,4), figsize=(14,6))
                    names.append(img["ids"])
            except Exception as e:
                print (e)
                continue
        
        if write_log == True: 
            #""" args.sample_size
            # stats_fi = "{0},{1},{2:.6f},{3:.6f},{4:.6f},{5:.6f},{6:.6f},{7:.6f},{8:.6f},{9:.6f},{10:.6f},{11:.6f},{12:.6f}".format(exp, ca, \
            #                                     np.mean(m_fi),np.mean(s_fi),np.mean(p_fi), \
            #                                     np.mean(pr_fi),np.mean(rc_fi),np.mean(fs_fi),np.mean(ac_fi),np.mean(sp_fi), \
            #                                     0, 0, np.mean(ces_fim))
            # stats_ro = "{0},{1},{2:.6f},{3:.6f},{4:.6f},{5:.6f},{6:.6f},{7:.6f},{8:.6f},{9:.6f},{10:.6f},{11:.6f},{12:.6f}".format(exp, ca, \
            #                                     np.mean(m_ro),np.mean(s_ro),np.mean(p_ro), \
            #                                     np.mean(pr_ro),np.mean(rc_ro),np.mean(fs_ro),np.mean(ac_ro),np.mean(sp_ro), \
            #                                     np.mean(dices),np.mean(ious), np.mean(ces_roi))

            titles = "Model,CE,lpips,mae,ssim,psnr"
            stats_fi = "{0},{1},{2:.6f},{3:.6f},{4:.6f},{5:.6f}".format(exp, ca, \
                                                np.mean(pdifs_fim),np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))
            stats_ro = "{0},{1},{2:.6f},{3:.6f},{4:.6f}".format(exp, ca, \
                                                np.mean(pdif_roi),np.mean(m_ro),np.mean(s_ro),np.mean(p_ro))
            
            # _, axes = plt.subplots(1,2, figsize=(16,5))
            # sns.heatmap(avg_conf_fim / args.sample_size, annot=True, fmt=".2f", cmap="hot",  ax = axes[0]) #vmin=0, vmax=1,
            # sns.heatmap(avg_conf_roi / args.sample_size, annot=True, fmt=".2f", cmap="hot",  ax = axes[1]) #vmin=0, vmax=1, xticklabels=real_ticks, yticklabels=pred_ticks,
            # plt.tight_layout()
            # #plt.savefig(output_dir + "avg_cm.png")
            # fig = plt.gcf()
            # fig = fig2img(fig)
            # fig.save(output_dir + "avg_cm.png")
            # np.save(output_dir + "avg_cm_f.png", avg_conf_fim)
            # np.save(output_dir + "avg_cm_r.png", avg_conf_roi)

            dict = {"name_exp" : titles,
                    args.exp_name + "-avg_fim" : stats_fi,
                    args.exp_name + "-avg_roi" : stats_ro}
            w = csv.writer(open(save_file, "a"))
            for key, val in dict.items(): w.writerow([key, val]) #"""
            print ("\n [!] -> Results saved in: {0} \n".format(save_file))

            if image_level:
                #
                dict = {"names": names, "mae_fi" : m_fi, "mae_roi" : m_ro, "lpips_fi" : pdifs_fim, "lpips_roi" : pdif_roi}
                dict = pd.DataFrame.from_dict(dict)
                dict.to_csv (os.path.splitext(save_file)[0] + "_im.csv", index=False)


##
## Image and stats saving in generation mode
## 
def generate_images_with_stats(args, dataloader, generator, epoch, shuffled = True, masked = False, image_level = False, \
                               output_dir = None, write_log = False, exp = "DEF", save_file = "Results/stats.csv"):
        #
        """Saves a generated sample from the validation set"""
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        # ce_m = torch.nn.L1Loss()
        # difference = True
        
        scaler = min_max_scaling(out_range = [-1,1])
        lpips_ = lpips.LPIPS(net=args.lpips_net)
        if args.cuda: lpips_.cuda()

        if args.sample_size == -1:
            args.sample_size = len(dataloader.test_generator)

        if output_dir == None: 
            output_dir = "%s/images/ep%s/" % (args.result_dir, epoch)

        if shuffled: 
            lucky = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
        else: 
            lucky = np.arange(0, args.sample_size)
        
        os.makedirs(output_dir+"imgs/", exist_ok = True)
        m_fi, s_fi, p_fi, m_ro, s_ro, p_ro = [], [], [], [], [], []
        ces_fim = []
        pdifs_fim, pdifs_roi = [], []
        names = []

        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            
            try:
                img = dataloader.test_generator[int(l)]

                real_in  = Variable(img["in" ].type(Tensor))  ; real_in  = real_in [None, :]
                real_out = Variable(img["out"].type(Tensor))  ; real_out = real_out[None, :]
                real_lab = Variable(img["label"].type(Tensor)); real_lab = real_lab[None, :]

                if args.model != "Mask_Pix2Pix" and args.model != "Mask_UNet" and \
                   args.model != "Mask_R_Pix2Pix" and args.model != "Mask_R_UNet" and \
                   args.model != "Mask_R_Pix2Pix_bloque" and args.model != "Mask_R_UNet_bloque":
                    fake_out = generator(real_in)
                else: 
                    fake_out = generator(real_in, real_lab)
                
                ###-----------
                ### Extraccion de los ROI
                ###-----------
                coords_ = extract_coords_roi(args, real_lab)
                roi_i = extract_roi_from_image(args, real_in, coords_)
                roi_r = extract_roi_from_image(args, real_out, coords_)
                roi_f = extract_roi_from_image(args, fake_out, coords_)
                roi_l = extract_roi_from_image(args, real_lab, coords_)
                
                roi_i = cv2.resize(roi_i, (64,64))
                roi_r = cv2.resize(roi_r, (64,64))
                roi_f = cv2.resize(roi_f, (64,64))
                roi_l = cv2.resize(roi_l, (64,64))
                 
                diffmap = abs(real_out.data - fake_out.data) 
                diffroi = abs(roi_r - roi_f) 
                img_sample = [real_in.data.cpu().numpy(), real_out.data.cpu().numpy(), fake_out.data.cpu().numpy(), real_lab.data.cpu().numpy(), \
                              roi_i, roi_r, roi_f, roi_l]
                diffmaps = [diffmap.cpu().numpy(), diffroi]
                
                ##---- Metrics -----##
                ##--- FIM ---##
                m_, s_, p_ = pixel_metrics(real_out.data.cpu().numpy(), fake_out.data.cpu().numpy(), masked=masked)
                m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)

                ##--- ROI ---##
                m_, s_, p_ = pixel_metrics(roi_r, roi_f)
                m_ro.append(m_), s_ro.append(s_), p_ro.append(p_)

                if image_level:
                    roi_r = torch.from_numpy(roi_r[np.newaxis,np.newaxis,:,:])
                    roi_f = torch.from_numpy(roi_f[np.newaxis,np.newaxis,:,:])
                    # roi_r = transforms.Resize(size=(64,64))(roi_r)
                    # roi_f = transforms.Resize(size=(64,64))(roi_f)
                    if args.cuda: 
                        roi_r = roi_r.cuda(); roi_f = roi_f.cuda()

                    pdif_fim = image_lpips(real_out, fake_out, scaler = scaler, lpips_ = lpips_)
                    pdif_roi = image_lpips(roi_r, roi_f, scaler = scaler, lpips_ = lpips_)
                    pdifs_fim.append(pdif_fim)
                    pdifs_roi.append(pdif_roi)
                
                save_images(img_sample, output_dir = output_dir + "imgs/%s.png" % (k), \
                            diffmap = diffmaps,image_ax = [0,1,2,3,5,6,7,8], diffmap_ax = [4,9], plot_shape = (2,5), figsize=(15,6))
                # save_images(img_sample, output_dir = output_dir + "imgs/%s.png" % (k), \
                #             diffmap = diffmaps,image_ax = [0,1,2,3,5,6,7,8], diffmap_ax = [4, 9], plot_shape = (2,5), figsize=(15,6))
                names.append(img["ids"])
            except Exception as e:
                print (e)
                continue
        
        if write_log == True: 
            #
            titles = "Model,CE,lpips,mae,ssim,psnr"
            stats_fi = "{0},{1:.6f},{2:.6f},{3:.6f},{4:.6f}".format(exp, \
                                                np.mean(pdifs_fim),np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))
            stats_ro = "{0},{1:.6f},{2:.6f},{3:.6f}".format(exp, \
                                                np.mean(pdif_roi),np.mean(m_ro),np.mean(s_ro),np.mean(p_ro))
            
            # stats_fi = "{0},{1:.6f},{2:.6f},{3:.6f},{4:.6f}".format(exp, \
            #                                     np.mean(pdifs_fim),np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))
            # stats_ro = "{0},{1:.6f},{2:.6f},{3:.6f}".format(exp, \
            #                                     np.mean(pdif_roi),np.mean(m_ro),np.mean(s_ro),np.mean(p_ro))
            
            dict = {"name_exp" : titles,
                    args.exp_name + "-avg_fim" : stats_fi,
                    args.exp_name + "-avg_roi" : stats_ro}
            w = csv.writer(open(save_file, "a"))
            for key, val in dict.items(): w.writerow([key, val]) #"""
            print ("\n [!] -> Results saved in: {0} \n".format(save_file))

            if image_level:
                #
                dict = {"names": names, "mae_fi" : m_fi, "lpips_fi" : pdifs_fim, "ssim_fi" : s_fi, "psnr_fi" : p_fi, 
                                        "mae_roi": m_ro, "lpips_roi": pdifs_roi, "ssim_roi": s_ro, "psnr_roi": p_ro }
                dict = pd.DataFrame.from_dict(dict)
                namefile = "im_" + os.path.splitext(save_file)[0][os.path.splitext(save_file)[0].rfind("/")+1:]
                namefile = os.path.splitext(save_file)[0][:os.path.splitext(save_file)[0].rfind("/")+1] + namefile + ".csv"
                dict.to_csv (namefile, index=False)



def extract_coords_roi(args, image):
    #
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy().squeeze()
    
    white_pixels = np.argwhere(image > 0)
    if white_pixels.size > 0:
        # Determinar las coordenadas mínimas y máximas
        min_y, min_x = white_pixels.min(axis=0)
        max_y, max_x = white_pixels.max(axis=0)
    else:
        min_y, min_x = image.shape[0]/2, image.shape[0]/2
        max_y, max_x = image.shape[1]/2, image.shape[1]/2
    
    # cropped_image_array = image[min_y:max_y + 1, min_x:max_x + 1]
    # cropped_image = Image.fromarray(cropped_image_array)
    return min_x, max_x, min_y, max_y


def extract_roi_from_image(args, image, coords):
    #
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy().squeeze()
    
    cropped_image_array = image[coords[2]:coords[3] + 1, coords[0]:coords[1] + 1]
    # cropped_image_array = image[min_y:max_y + 1, min_x:max_x + 1]
    # cropped_image = Image.fromarray(cropped_image_array)
    return cropped_image_array


##
## Generate and save images
## 
def generate_images(args, dataloader, generator, epoch, shuffled = True, output_dir = None):
        #
        """Saves a generated sample from the validation set"""
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        difference = True
        
        scaler = min_max_scaling(out_range = [-1,1])
        
        if args.sample_size == -1:
            args.sample_size = len(dataloader.test_generator)

        if output_dir == None: 
            output_dir = "%s/images/ep%s/" % (args.result_dir, epoch)

        if shuffled: 
            lucky = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
        else: 
            lucky = np.arange(0, args.sample_size)
        
        os.makedirs(output_dir+"generated/img", exist_ok = True)
        os.makedirs(output_dir+"generated/roi", exist_ok = True)

        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            
            try:
                name = dataloader.test_generator.targs[int(l)].split('/')
                del name[-2]
                name = '-'.join(name[-2:])
                img = dataloader.test_generator[int(l)]
                real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
                real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
                roi_in  = Variable(img["roi_in" ].type(Tensor)); roi_in = roi_in[None, :]
                roi_out = Variable(img["roi_out"].type(Tensor)); roi_out = roi_out[None, :]

                fake_out = generator(real_in)
                roi_fake = extract_roi_from_points(fake_out.cpu().detach().numpy(), img["pt_roi_out"] )
                
                im = Image.fromarray(fake_out.detach().cpu().numpy().squeeze())
                im.save(output_dir+"generated/img/"+name)
                im = Image.fromarray(roi_fake.squeeze())
                im.save(output_dir+"generated/roi/"+name)
                
            except Exception as e:
                print (e)
                continue


##
## Image and stats saving in generation mode
## 
def pixel_metrics_from_path (args, dataloader, epoch, shuffled = True, masked = False, \
                             output_dir = None, write_log = False, save_file = "Results/stats.csv"):
        #
        """Compute metrics from files"""
        difference = True

        if args.sample_size == -1:
            args.sample_size = len(dataloader.test_generator)

        if output_dir == None: 
            output_dir = "%s/images/ep%s/" % (args.result_dir, epoch)

        if shuffled: 
            lucky = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
        else: 
            lucky = np.arange(0, args.sample_size)
        
        m_fi, s_fi, p_fi, m_ro, s_ro, p_ro = [], [], [], [], [], []
        names, ids = [], []

        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            
            try:
                img = dataloader.test_generator[int(l)]
                # real_in  = img["in" ].numpy().squeeze() #;     real_in  = real_in [None, :]
                real_out = img["out" ].numpy().squeeze() #;    real_out = real_out[None, :]
                # roi_in   = img["roi_in" ].numpy().squeeze() #; roi_in   = roi_in  [None, :]
                roi_out  = img["roi_out"].numpy().squeeze() #; roi_out  = roi_out [None, :]

                id_ = dataloader.test_generator.targs[int(l)].split('/')
                del id_[-2]
                id_ = '-'.join(id_[-2:])

                fake_out = output_dir + '/generated/img/{0}'.format(id_)
                fake_out = np.array(Image.open(fake_out).convert('F'))
                roi_fake = output_dir + '/generated/roi/{0}'.format(id_)
                roi_fake = np.array(Image.open(roi_fake).convert('F'))
                
                # fake_out = generator(real_in)
                # roi_fake = extract_roi_from_points(fake_out.cpu().detach().numpy(), img["pt_roi_out"] )
                
                ##---- Metrics -----##
                ##--- FIM ---##
                m_, s_, p_ = pixel_metrics(real_out, fake_out, masked=masked)
                m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)

                ##--- ROI ---##
                m_, s_, p_ = pixel_metrics(roi_out, roi_fake)
                m_ro.append(m_), s_ro.append(s_), p_ro.append(p_)
                
                names.append(img["ids"])
                ids.append  (id_)
            except Exception as e:
                print (e)
                continue
        
        if write_log == True: 
            #
            # titles = "Model,CE,lpips,mae,ssim,psnr"
            # stats_fi = "{0},{1},{2:.6f},{3:.6f},{4:.6f},{5:.6f}".format(exp, norm_, \
            #                                     np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))
            # stats_ro = "{0},{1},{2:.6f},{3:.6f},{4:.6f}".format(exp, norm_, \
            #                                     np.mean(pdif_roi),np.mean(m_ro),np.mean(s_ro),np.mean(p_ro))
            
            # dict = {"name_exp" : titles,
            #         args.exp_name + "-avg_fim" : stats_fi,
            #         args.exp_name + "-avg_roi" : stats_ro}
            # w = csv.writer(open(save_file, "a"))
            # for key, val in dict.items(): w.writerow([key, val]) #"""

            dict = {"names": names, "ids": ids, "mae_fi" : m_fi, "mae_roi" : m_ro, 
                    "ssim_fi" : s_fi, "ssim_roi" : s_ro, "psnr_fi" : p_fi, "psnr_roi" : p_ro}
            dict = pd.DataFrame.from_dict(dict)
            namefile = "im_" + os.path.splitext(save_file)[0][os.path.splitext(save_file)[0].rfind("/")+1:]
            namefile = os.path.splitext(save_file)[0][:os.path.splitext(save_file)[0].rfind("/")+1] + namefile + ".csv"
            dict.to_csv (namefile, index=False)

            print ("\n [!] -> Results saved in: {0} \n".format(save_file))



##
## Image saving during training
## 
def sample_images(args, dataloader, generator, epoch, difference = True, output_dir = None, shuffled = True, write_log = False, logger=None):
        #
        """Saves a generated sample from the validation set"""
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor

        if args.sample_size == -1:
            args.sample_size = len(dataloader.test_generator)

        if output_dir == None: 
            output_dir = "%s/images/ep%s/" % (args.result_dir, epoch)

        if shuffled: 
            lucky = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
        else: 
            lucky = np.arange(0, args.sample_size)
        
        os.makedirs(output_dir, exist_ok = True)
        diff_imgs, diff_rois, m_fi, s_fi, p_fi, m_ro, s_ro, p_ro = [], [], [], [], [], [], [], []
        list_im, list_tic = [], []

        generator.eval()

        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            img = dataloader.test_generator[int(l)]
            real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
            real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
            real_lab = Variable(img["label"].type(Tensor)); real_lab = real_lab[None, :]
            # roi_in  = Variable(img["roi_in" ].type(Tensor)); roi_in = roi_in[None, :]
            # roi_out = Variable(img["roi_out"].type(Tensor)); roi_out = roi_out[None, :]

            # # GAN loss 
            if args.model != "Mask_Pix2Pix" and args.model != "Mask_UNet" and args.model != "Mask_R_Pix2Pix" and args.model != "Mask_R_UNet" and args.model != "Mask_R_Pix2Pix_bloque" and args.model != "Mask_R_UNet_bloque":
                fake_out = generator(real_in)
            else: 
                fake_out = generator(real_in, real_lab)
            # roi_fake = extract_roi_from_points(fake_out.cpu().detach().numpy(), img["pt_roi_out"] )

            if difference:
                diffmap = abs(real_out.data - fake_out.data) 
                # diffroi = abs(roi_out.data.cpu().numpy() - roi_fake[np.newaxis,:,:,:]) 
                # img_sample = [real_in.data.cpu().numpy(), real_out.data.cpu().numpy(), fake_out.data.cpu().numpy(), \
                #               roi_in.data.cpu().numpy(), roi_out.data.cpu().numpy(), roi_fake[np.newaxis,:,:,:]]
                img_sample = [real_in.data.cpu().numpy(), real_out.data.cpu().numpy(), fake_out.data.cpu().numpy(), real_lab.data.cpu().numpy()]
                diffmaps = [diffmap.cpu().numpy()]

                if logger: 
                    ## plot all images
                    im = save_images(img_sample, output_dir = output_dir + "%s.png" % (k), image_ax = [0,1,2,3], \
                            diffmap = diffmaps, diffmap_ax = [4], plot_shape = (1,5), figsize=(20,4), return_image = True)
                    list_im.append(im)
                    
                    # tic = time_intensity_training(roi_in, roi_out, roi_fake)
                    # list_tic.append (tic)

                else: 
                    save_images(img_sample, output_dir = output_dir + "%s.png" % (k), \
                            diffmap = diffmaps, diffmap_ax = [3, 7], plot_shape = (2,4), figsize=(14,6))
                
                ##---- Metrics -----
                m_, s_, p_ = pixel_metrics(real_out.data.cpu().numpy(), fake_out.data.cpu().numpy())
                m_fi.append(m_), s_fi.append(s_), p_fi.append(p_)

                # m_, s_, p_ = pixel_metrics((roi_out.data.cpu().numpy()+1)/2, (roi_fake[np.newaxis,:,:,:]+1)/2)
                # m_ro.append(m_), s_ro.append(s_), p_ro.append(p_)

            else:
                img_sample = torch.cat((real_in.data, real_out.data, fake_out.data), -1)
                save_image(img_sample, output_dir + "%s.png" % (k), normalize=True)
        
        if logger: 
            logger.write_images(list_im, epoch)
            # logger.write_images(list_tic, epoch, custom_name = "TIC")

        if write_log == True: 
            #"""
            stats_fi = "{0:.4f}, {1:.4f}, {2:.4f}".format(np.mean(m_fi),np.mean(s_fi),np.mean(p_fi))
            # stats_ro = "{0:.4f}, {1:.4f}, {2:.4f}".format(np.mean(m_ro),np.mean(s_ro),np.mean(p_ro))
            dict = {args.exp_name + "-avg_fim" : stats_fi}
                    # args.exp_name + "-avg_roi" : stats_ro}
            w = csv.writer(open("Results/stats.csv", "a"))
            for key, val in dict.items(): w.writerow([key, val]) #"""
            print ("\n [!] -> Results saved in: Results/stats.csv \n")

         
##
## Compute CE ratio
## 
def ce_ratio (real_in, real_out, fake_out, output_file="", save_fig = True, cuda = False, tolerance = 0.2):
    #
    ##-------- Second plot --------##
    if cuda: 
        # real_in_arr  = np.squeeze(real_in.cpu().numpy()) +1
        # real_out_arr = np.squeeze(real_out.cpu().numpy()) +1
        # fake_out_arr = np.squeeze(fake_out.detach().cpu().numpy()) +1
        
        real_inc = (torch.round(real_in+1, decimals=3)+tolerance < torch.round(real_out+1, decimals=3)) * 3
        fake_inc = (torch.round(real_in+1, decimals=3)+tolerance < torch.round(fake_out+1, decimals=3)) * 3

        real_eql = (torch.round(real_in+1, decimals=3)-tolerance <= torch.round(real_out+1, decimals=3)) * 1
        fake_eql = (torch.round(real_in+1, decimals=3)-tolerance <= torch.round(fake_out+1, decimals=3)) * 1

        real_equ = (torch.round(real_in+1, decimals=3)+tolerance >= torch.round(real_out+1, decimals=3)) * 1
        fake_equ = (torch.round(real_in+1, decimals=3)+tolerance >= torch.round(fake_out+1, decimals=3)) * 1

        real_eq = (real_eql + real_equ)-1
        fake_eq = (fake_eql + fake_equ)-1

        real_dec = (torch.round(real_in, decimals=3)-tolerance > torch.round(real_out, decimals=3)) * 2
        fake_dec = (torch.round(real_in, decimals=3)-tolerance > torch.round(fake_out, decimals=3)) * 2
    else: 
        real_in_arr  = np.squeeze(real_in.cpu().numpy()) +1
        real_out_arr = np.squeeze(real_out.cpu().numpy()) +1
        fake_out_arr = np.squeeze(fake_out.detach().cpu().numpy()) +1

        real_inc = (np.around(real_in_arr, decimals=3)+tolerance < np.around(real_out_arr, decimals=3)) * 3
        fake_inc = (np.around(real_in_arr, decimals=3)+tolerance < np.around(fake_out_arr, decimals=3)) * 3

        real_eql = (np.around(real_in_arr, decimals=3)-tolerance <= np.around(real_out_arr, decimals=3)) * 1
        fake_eql = (np.around(real_in_arr, decimals=3)-tolerance <= np.around(fake_out_arr, decimals=3)) * 1

        real_equ = (np.around(real_in_arr, decimals=3)+tolerance >= np.around(real_out_arr, decimals=3)) * 1
        fake_equ = (np.around(real_in_arr, decimals=3)+tolerance >= np.around(fake_out_arr, decimals=3)) * 1

        real_eq = (real_eql + real_equ)-1
        fake_eq = (fake_eql + fake_equ)-1

        real_dec = (np.around(real_in_arr, decimals=3)-tolerance > np.around(real_out_arr, decimals=3)) * 2
        fake_dec = (np.around(real_in_arr, decimals=3)-tolerance > np.around(fake_out_arr, decimals=3)) * 2


    if save_fig: 
        ctags = ["same", "lower", "higher"]
        f, axes = plt.subplots(1,2, figsize = (10,5))
        axes[0].imshow((real_inc + real_eq + real_dec)-1, cmap="seismic")
        # axes[1].imshow(fake_inc + fake_eq + fake_dec, cmap="coolwarm")
        f = f.colorbar(axes[1].imshow((fake_inc + fake_eq + fake_dec)-1, cmap="seismic"), ax=axes.tolist(), ticks=[0, 1, 2], fraction=0.021)
        f.ax.set_yticklabels(ctags) 

        axes[0].set_axis_off(), axes[1].set_axis_off()
        axes[0].set_title("Real CE"), axes[1].set_title("Generated CE")
        #plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close(), plt.clf()

    return (real_inc, real_eq, real_dec), (fake_inc, fake_eq, fake_dec)


##
## Image and stats saving in generation mode
## 
def generate_ce_metrics(args, dataloader, generator, epoch, shuffled = True, \
                               output_dir = None, write_log = False, exp = "DEF", ca="CA", save_file = "Results/ce_stats.csv"):
        #
        """Saves a generated sample from the validation set"""
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        ce_m = torch.nn.L1Loss()
        difference = True

        if args.sample_size == -1:
            args.sample_size = len(dataloader.test_generator)

        if output_dir == None: 
            output_dir = "%s/images/ep%s/" % (args.result_dir, epoch)

        if shuffled: 
            lucky = np.random.randint(0, len(dataloader.test_generator), args.sample_size)
        else: 
            lucky = np.arange(0, args.sample_size)
        
        os.makedirs(output_dir+"ce_fi/", exist_ok = True) 
        os.makedirs(output_dir+"ce_up/", exist_ok = True)
        os.makedirs(output_dir+"ce_upr/", exist_ok = True)
        ious_h, ious_s, ious_l, ious_t = [], [], [], []
        ious_rh, ious_rs, ious_rl, ious_rt = [], [], [], []

        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            
            try:
                img = dataloader.test_generator[int(l)]
                real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
                real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
                roi_in  = Variable(img["roi_in" ].type(Tensor)); roi_in = roi_in[None, :]
                roi_out = Variable(img["roi_out"].type(Tensor)); roi_out = roi_out[None, :]

                fake_out = generator(real_in)
                roi_fake = extract_roi_from_points(fake_out.cpu().detach().numpy(), img["pt_roi_out"] )
                roi_fake = Variable(torch.from_numpy(roi_fake).type(Tensor))

                real_ca = real_in - real_out
                fake_ca = real_in - fake_out

                real_rca = roi_in - roi_out
                fake_rca = roi_in - roi_fake

                ##-------- First plot --------##
                list_imgs = [real_in, fake_out, fake_ca, real_out, real_ca, torch.zeros([10,10]), roi_in, roi_fake, fake_rca, roi_out, real_rca, torch.zeros([10,10])]
                list_titles = ["Input", "Fake", "Fake CE", "Out", "Out CE", "Diff", "", "", "", "", "", ""]

                num_figs = len(list_imgs)
                _, axes = plt.subplots(2,int(num_figs/2), figsize=(18,8))
                axes = axes.ravel()

                fig_idx = np.arange (num_figs)

                for i, image, title in zip(fig_idx, list_imgs, list_titles): 
                    #
                    #print (image.detach().cpu().numpy().min(), image.detach().cpu().numpy().max(), )
                    axes[i].imshow(np.squeeze(image.detach().cpu().numpy()), cmap="gray") #, vmin=-1, vmax=1
                    axes[i].set_title(title); axes[i].set_axis_off()

                diffmap = abs(np.squeeze(real_ca.detach().cpu().numpy()) - np.squeeze(fake_ca.detach().cpu().numpy())) 
                diffroi = abs(np.squeeze(real_rca.detach().cpu().numpy()) - np.squeeze(fake_rca.detach().cpu().numpy()))

                sns.heatmap(np.squeeze(diffmap), cmap = "coolwarm", ax=axes[5]) #, vmin=0, vmax=1
                sns.heatmap(np.squeeze(diffroi), cmap = "coolwarm", ax=axes[11]) #, vmin=0, vmax=1

                plt.tight_layout()
                plt.savefig(output_dir + "ce_fi/%s.png" % (k), dpi=150)
                plt.close(), plt.clf()

                ## CE ratio
                real_enh, fake_enh = ce_ratio (real_in, real_out, fake_out, output_file = output_dir + "ce_up/%s.png" % (k))
                (real_inc, real_eq, real_dec) = real_enh
                (fake_inc, fake_eq, fake_dec) = fake_enh
                
                ca_h_matches = IoU_between_masks  ((real_inc>0.5)*1, (fake_inc>0.5)*1, k = 1.0)
                ca_s_matches = IoU_between_masks  ((real_eq>0.5)*1, (fake_eq>0.5)*1, k = 1.0)
                ca_l_matches = IoU_between_masks  ((real_dec>0.5)*1, (fake_dec>0.5)*1, k = 1.0)

                ca_matches = ( (fake_inc + fake_eq + fake_dec)-1 == (real_inc + real_eq + real_dec)-1 ) * 1

                ious_h.append(ca_h_matches), ious_s.append(ca_s_matches), ious_l.append(ca_l_matches)
                ious_t.append(ca_matches)

                ## CE ratio - ROIs
                real_enh, fake_enh = ce_ratio (roi_in, roi_out, roi_fake, output_file = output_dir + "ce_upr/%s.png" % (k))
                (real_inc, real_eq, real_dec) = real_enh
                (fake_inc, fake_eq, fake_dec) = fake_enh
                
                ca_h_matches = IoU_between_masks  ((real_inc>0.5)*1, (fake_inc>0.5)*1, k = 1.0)
                ca_s_matches = IoU_between_masks  ((real_eq>0.5)*1, (fake_eq>0.5)*1, k = 1.0)
                ca_l_matches = IoU_between_masks  ((real_dec>0.5)*1, (fake_dec>0.5)*1, k = 1.0)

                ca_matches = ( (fake_inc + fake_eq + fake_dec)-1 == (real_inc + real_eq + real_dec)-1 ) * 1

                ious_rh.append(ca_h_matches), ious_rs.append(ca_s_matches), ious_rl.append(ca_l_matches)
                ious_rt.append(ca_matches)

            except Exception as e:
                print (e)
                continue
        
        ious_h, ious_s, ious_l, ious_t
        if write_log == True: 
            #""" args.sample_size
            stats_fi = "{0},{1},{2:.6f},{3:.6f},{4:.6f},{5:.6f}".format(exp, ca, \
                                                np.mean(ious_h),np.mean(ious_s),np.mean(ious_l),np.mean(ious_t))
            stats_ro = "{0},{1},{2:.6f},{3:.6f},{4:.6f},{5:.6f}".format(exp, ca, \
                                                np.mean(ious_rh),np.mean(ious_rs),np.mean(ious_rl),np.mean(ious_rt))

            dict = {args.exp_name + "avg_fim" : stats_fi,
                    args.exp_name + "avg_roi" : stats_ro}
            w = csv.writer(open(save_file, "a"))
            for key, val in dict.items(): w.writerow([key, val]) #"""
            print ("\n [!] -> Results saved in: {0} \n".format(save_file))


##
## Image and stats saving in generation mode
## 
def cer_maps(batch_real_in, batch_real_out, batch_fake_out, cuda = True, return_matches = False):
        #
        """Computes the CER maps across images in a batch"""
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        if return_matches: 
            h_matches, s_matches, l_matches = [], [], []

        for real_in, real_out, fake_out in zip(batch_real_in, batch_real_out, batch_fake_out):
            
            try:
                ## CE ratio
                real_enh, fake_enh = ce_ratio (real_in, real_out, fake_out, cuda = cuda, save_fig = False)
                (real_inc, real_eq, real_dec) = real_enh
                (fake_inc, fake_eq, fake_dec) = fake_enh
                
                if return_matches: 
                    h_match = IoU_between_masks  ((real_inc>0.5)*1, (fake_inc>0.5)*1, k = 1.0, cuda = cuda)
                    s_match = IoU_between_masks  ((real_eq>0.5)*1, (fake_eq>0.5)*1, k = 1.0, cuda = cuda)
                    l_match = IoU_between_masks  ((real_dec>0.5)*1, (fake_dec>0.5)*1, k = 1.0, cuda = cuda)
                    h_matches.append(h_match)
                    s_matches.append(s_match)
                    l_matches.append(l_match)

            except Exception as e:
                print (e)
                continue
        
        if return_matches:
            return (real_inc, real_eq, real_dec), (fake_inc, fake_eq, fake_dec), (np.mean(h_matches), np.mean(s_matches), np.mean(l_matches))
        else:
            return (real_inc, real_eq, real_dec), (fake_inc, fake_eq, fake_dec)


##
## Image and stats saving in generation mode
## 
def generate_time_intensity_reference(args, pre_early, early_late, generator, epoch, shuffled = False, \
                                      point_max=False, draw_patches = False, \
                                      output_dir = None, save_file = "Results/tic_stats.csv"):
        #
        """Saves a generated sample from the validation set"""
        Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
        if point_max: draw_patches = True
        
        if args.sample_size == -1:
            args.sample_size = len(pre_early.test_generator)
        
        if output_dir == None: 
            output_dir = "%s/images/ep%s/" % (args.result_dir, epoch)

        if shuffled: 
            lucky = np.random.randint(0, len(pre_early.test_generator), args.sample_size)
        else: 
            lucky = np.arange(0, args.sample_size)
        
        os.makedirs(output_dir+"time_i2/", exist_ok = True) 
        save_dict = {"name": [], "pre": [], "early": [], "late": [], "gen": []}
        
        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            
            try:
                img_ = pre_early.test_generator[int(l)]
                img = early_late.test_generator[int(l)]

                real_ref = Variable(img_["in"].type(Tensor)); real_ref = real_ref[None, :]
                real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
                real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
                roi_ref  = Variable(img_["roi_in" ].type(Tensor)); roi_ref = roi_ref[None, :]
                roi_in  = Variable(img["roi_in" ].type(Tensor)); roi_in = roi_in[None, :]
                roi_out = Variable(img["roi_out"].type(Tensor)); roi_out = roi_out[None, :]

                fake_out = generator(real_in)
                roi_fake = extract_roi_from_points(fake_out.cpu().detach().numpy(), img["pt_roi_out"] )
                roi_fake = Variable(torch.from_numpy(roi_fake).type(Tensor))

                if draw_patches:
                    max_inds = torch.where(roi_in[0,0,10:-10, 10:-10] == roi_in[0,0,10:-10, 10:-10].max())
                    # ct_x, ct_y = int(max_inds[0]), int(max_inds[1])
                    # max_inds = torch.where(roi_in == roi_in.max())
                    ct_x, ct_y = int(max_inds[-2] + 10), int(max_inds[-1] + 10)
                    p_i = (img["pt_roi_out"][0][1], img["pt_roi_out"][1][1])
                    p_e = (img["pt_roi_out"][0][0], img["pt_roi_out"][1][0]) 

                    real_ref = draw_roi_tensor(real_ref, p_i, p_e)
                    real_in  = draw_roi_tensor(real_in,  p_i, p_e)
                    fake_out = draw_roi_tensor(fake_out, p_i, p_e)
                    real_out = draw_roi_tensor(real_out, p_i, p_e)

                    roi_ref  = draw_roi_tensor(roi_ref,  (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))
                    roi_in   = draw_roi_tensor(roi_in,   (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))
                    roi_fake = draw_roi_tensor(roi_fake, (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))
                    roi_out  = draw_roi_tensor(roi_out,  (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))
                
                ##-------- First plot --------##
                list_imgs = [real_ref, real_in, fake_out, real_out, None, roi_ref, roi_in, roi_fake, roi_out, None]
                list_titles = ["Reference", "Input", "Fake", "Out", "Time intensity", "", "", "", ""]

                num_figs = len(list_imgs)
                _, axes = plt.subplots(2,int(num_figs/2), figsize=(int(num_figs/2)*5,int(num_figs/2)*2))
                axes = axes.ravel()

                fig_idx = np.arange (num_figs)

                for i, image, title in zip(fig_idx, list_imgs, list_titles): 
                    #
                    if not image is None : # != None: 
                        if isinstance(image, Tensor): 
                            axes[i].imshow(np.squeeze(image.detach().cpu().numpy()), cmap="gray") #, vmin=-1, vmax=1
                        else: 
                            axes[i].imshow(np.squeeze(image), cmap="gray") #, vmin=-1, vmax=1
                        axes[i].set_title(title); axes[i].set_axis_off()
                
                if point_max:
                    roi_ref  = roi_ref [ct_x-1 : ct_x+2, ct_y-1 : ct_y+2] #Variable(img_["in"].type(Tensor))
                    roi_in   = roi_in  [ct_x-1 : ct_x+2, ct_y-1 : ct_y+2]
                    roi_fake = roi_fake[ct_x-1 : ct_x+2, ct_y-1 : ct_y+2]
                    roi_out  = roi_out [ct_x-1 : ct_x+2, ct_y-1 : ct_y+2]
                    
                
                time_intensity_testing(roi_ref, roi_in, roi_out, roi_fake, name=img["ids"], ax=axes[4], save_dict = save_dict)
                axes[9].remove()

                plt.tight_layout()
                plt.savefig(output_dir + "time_i2/%s.png" % (k), dpi=250)
                plt.close(), plt.clf()
                
            except Exception as e:
                print (e)
                continue
        
        save_dict = pd.DataFrame.from_dict(save_dict)
        save_dict.to_csv(save_file, index=False)
        print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")


##
## Compute time intensity during training (just 2 points)
## 
def time_intensity_testing(roi_ref, roi_in, roi_out, roi_fake, name="", ax=None, save_dict=None):
    # Compute the TIC trend to the logs
    if isinstance(roi_ref, np.ndarray):
        m_roi_ref  = np.mean(roi_ref)
        m_roi_in  = np.mean(roi_in)
        m_roi_out = np.mean(roi_out)
        m_roi_fake = np.mean(roi_fake)
    else:
        m_roi_ref  = np.mean(roi_ref.detach().cpu().numpy())
        m_roi_in  = np.mean(roi_in.detach().cpu().numpy())
        m_roi_out = np.mean(roi_out.detach().cpu().numpy())
        m_roi_fake = np.mean(roi_fake.detach().cpu().numpy())

    if ax == None: 
        fig, ax = plt.subplots (figsize=(5,5)) 
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Pixel Intensity")

    ax.plot(range (1,4), [m_roi_ref, m_roi_in, m_roi_out],  label="Real", linestyle='-' , linewidth = 2)
    ax.plot(range (1,4), [m_roi_ref, m_roi_in, m_roi_fake], label="Gen ", linestyle='--', linewidth = 2)

    inc = (m_roi_out - m_roi_in)/m_roi_in
    threshold = (abs(m_roi_in - m_roi_out) * 0.1) / inc
    ax.hlines(y = [m_roi_in + threshold, m_roi_in - threshold], xmin=[1, 1], xmax=[3, 3], linestyle='--', linewidth = 1, alpha = 0.5, color='gray')
    # ax.hlines(y = m_roi_in - threshold, xmin=1, xmax=4, linestyle='--', linewidth = 2, color='gray')
    ax.set_title("Time Intensity Curve")
    ax.legend()
    ax.set_xticks(range (1,4))

    if save_dict:
        save_dict["name"].append(name)
        save_dict["pre"].append(m_roi_ref)
        save_dict["early"].append(m_roi_in)
        save_dict["late"].append(m_roi_out)
        save_dict["gen"].append(m_roi_fake)
    
    # plot_as_image = fig2img(fig)
    # return ax



##
## Compute time intensity during training (just 2 points)
## 
def time_intensity_maps(im_ref, im_in, im_out, cuda = False):
    # Compute the TIC 
    if not cuda: 
        im_ref = im_ref.detach().cpu().numpy()
        im_in = im_in.detach().cpu().numpy()
        im_out = im_out.detach().cpu().numpy()

    thresholds = (abs(im_in.mean() - im_ref.mean()) + 0.0001) * 0.15
    # thresholds = (abs(im_in - im_ref) + 0.01) * 0.1
    persist = im_out > (im_in+thresholds)
    plateau_h = im_out <= (im_in + thresholds)
    plateau_l = im_out >= (im_in - thresholds)
    plateau = plateau_h * plateau_l
    washout = im_out < (im_in - thresholds)

    return persist, plateau, washout


##
## Extract random patches from images
## 
def generate_time_intensity_patches(args, pre_early, early_late, generator, epoch, shuffle = False, \
                                    point_max=False, draw_patches = False, \
                                    output_dir = "Results/", save_file = "Results/patches_meta.csv", points = None):
    #
    """ Extract random patches from images """
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    
    if point_max: draw_patches = True

    if args.sample_size == -1:
        args.sample_size = len(pre_early.test_generator)
    
    if output_dir == None: 
        output_dir = "%s/patches/ep%s/" % (args.result_dir, epoch)

    if shuffle: 
        lucky = np.random.randint(0, len(pre_early.test_generator), args.sample_size)
    else: 
        lucky = np.arange(0, args.sample_size)
    
    os.makedirs(output_dir+"patches/{0}/".format(args.random_roi_size), exist_ok = True) 
    save_dict = {"name": [], "pre": [], "early": [], "late": [], "gen": []}

    points = pd.read_csv(points)
    
    np.random.seed(args.random_patch_seed)
    for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
        for npt in range(args.random_patches_per_image):
            # if k == 5: break
            try:
                idx = npt+(k*args.random_patches_per_image) #k+(npt*len(lucky))
                p_x = (points['x0'].iloc[idx], points['x1'].iloc[idx])
                p_y = (points['y0'].iloc[idx], points['y1'].iloc[idx])
                img_ = pre_early.test_generator[int(l)]
                img = early_late.test_generator[int(l)]

                real_ref = Variable(img_["in"].type(Tensor)); real_ref = real_ref[None, :]
                real_in  = Variable(img["in" ].type(Tensor)); real_in = real_in[None, :]
                real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
                roi_ref  = Variable(img_["roi_in" ].type(Tensor)); roi_ref = roi_ref[None, :]
                roi_in  = Variable(img["roi_in" ].type(Tensor)); roi_in = roi_in[None, :]
                roi_out = Variable(img["roi_out"].type(Tensor)); roi_out = roi_out[None, :]

                fake_out = generator(real_in)
                # roi_fake = extract_roi_from_points(fake_out.cpu().detach().numpy(), img["pt_roi_out"] )
                roi_ref, roi_in, roi_out, roi_fake, axes = extract_random_roi(real_ref, real_in, real_out, fake_out, \
                                            roi_size=args.random_roi_size, threshold = 0.5, draw=draw_patches, \
                                            p_x = p_x, p_y = p_y ) #save_image=output_dir+"patches/{0}/{1}.png".format(args.random_roi_size, k)
                
                if draw_patches:
                    max_inds = torch.where(roi_in == roi_in.max())
                    ct_x, ct_y = int(max_inds[-2] ), int(max_inds[-1] )

                    if ct_x == args.random_roi_size-1: ct_x -= 1
                    if ct_y == args.random_roi_size-1: ct_y -= 1

                    if ct_x == 0: ct_x = 1
                    if ct_y == 0: ct_y = 10
                    
                    p_i = (img["pt_roi_out"][0][1], img["pt_roi_out"][1][1])
                    p_e = (img["pt_roi_out"][0][0], img["pt_roi_out"][1][0]) 

                    real_ref = draw_roi_tensor(real_ref, p_i, p_e)
                    real_in  = draw_roi_tensor(real_in,  p_i, p_e)
                    fake_out = draw_roi_tensor(fake_out, p_i, p_e)
                    real_out = draw_roi_tensor(real_out, p_i, p_e)

                    roi_ref  = draw_roi_tensor(roi_ref,  (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))
                    roi_in   = draw_roi_tensor(roi_in,   (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))
                    roi_fake = draw_roi_tensor(roi_fake, (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))
                    roi_out  = draw_roi_tensor(roi_out,  (ct_x-2, ct_x+2), (ct_y-2, ct_y+2))

                    axes[5].imshow(roi_ref,  cmap='gray')
                    axes[6].imshow(roi_in,   cmap='gray')
                    axes[7].imshow(roi_out,  cmap='gray')
                    axes[8].imshow(roi_fake, cmap='gray')
                
                if point_max:
                    roi_ref  = roi_ref [ct_x-1 : ct_x+2, ct_y-1: ct_y+2] #Variable(img_["in"].type(Tensor))
                    roi_in   = roi_in  [ct_x-1 : ct_x+2, ct_y-1: ct_y+2]
                    roi_fake = roi_fake[ct_x-1 : ct_x+2, ct_y-1: ct_y+2]
                    roi_out  = roi_out [ct_x-1 : ct_x+2, ct_y-1: ct_y+2]
                    

                time_intensity_testing(roi_ref, roi_in, roi_out, roi_fake, name=img["ids"], ax=axes[4], save_dict = save_dict)
                axes[4].set_axis_on(), axes[9].remove()

                plt.tight_layout()
                plt.savefig(output_dir + "patches/{0}/{1}.png".format(args.random_roi_size, k+(npt*len(lucky))), dpi=250)
                plt.close(), plt.clf()
                
            except Exception as e:
                print (e)
                continue

            if k % 100 == 0:
                temp_dict = pd.DataFrame.from_dict(save_dict)
                temp_dict.to_csv(save_file, index=False)
    
    save_dict = pd.DataFrame.from_dict(save_dict)
    save_dict.to_csv(save_file, index=False)
    print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")



def generate_time_intensity_points(args, pre_early, early_late, shuffle = False, \
                                   output_dir = "Results/", save_file = "patches_meta.csv"):
    #
    """ Extract random patches from images """
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
    
    if args.sample_size == -1:
        args.sample_size = len(pre_early.test_generator)
    
    if output_dir == None: 
        output_dir = "%s/patches/" % (args.result_dir)

    if shuffle: 
        lucky = np.random.randint(0, len(pre_early.test_generator), args.sample_size)
    else: 
        lucky = np.arange(0, args.sample_size)
    
    os.makedirs(output_dir, exist_ok = True) 
    save_dict = {"x0": [], "x1": [], "y0": [], "y1": []}

    np.random.seed(args.random_patch_seed)
    for npt in range(args.random_patches_per_image):
        for k, l in tqdm(enumerate(lucky), ncols=100, total=len(lucky)):
            # if k == 5: break
            try:
                img_ = pre_early.test_generator[int(l)]
                img = early_late.test_generator[int(l)]

                real_ref = Variable(img_["in"].type(Tensor)); real_ref = real_ref[None, :]
                real_in  = Variable(img["in" ].type(Tensor)); real_in  = real_in [None, :]
                real_out = Variable(img["out"].type(Tensor)); real_out = real_out[None, :]
                
                p_i = (img["pt_roi_out"][0][1], img["pt_roi_out"][1][1])
                p_e = (img["pt_roi_out"][0][0], img["pt_roi_out"][1][0]) 
                
                p_x, p_y = extract_roi_from_roi_points( min(p_i), min(p_e), \
                                                        image_size = args.image_size, roi_size=args.random_roi_size, )
                
                save_dict["x0"].append(p_x[0]), save_dict["x1"].append(p_x[1])
                save_dict["y0"].append(p_y[0]), save_dict["y1"].append(p_y[1])
                
            except Exception as e:
                print (e)
                continue
    
    save_dict = pd.DataFrame.from_dict(save_dict)
    save_dict.to_csv(save_file, index=False)
    print ("\n" + Fore.GREEN + "[✓] -> Done!" + Fore.RESET + "\n\n")


def extract_roi_from_roi_points ( px, py, image_size = 256, roi_size = 32 ):
    #
    if py <= 128: 
        xt, yt = px , image_size - py #+ np.random.randint(0, 15) + np.random.randint(0, 7)
    else:
        xt, yt = px , image_size - py # - np.random.randint(0, 15) - np.random.randint(0, 7)
    
    p_x, p_y = (xt - int(roi_size/2), xt + int(roi_size/2)), (yt - int(roi_size/2), yt + int(roi_size/2))
    return p_x, p_y
