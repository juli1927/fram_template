

import os
import cv2
import numpy as np
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')

# Structural Similarity Measure
from sklearn.metrics import confusion_matrix
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from skimage import filters
from skimage.morphology import square, dilation

import torch

import warnings
warnings.filterwarnings('ignore')


def fig2img(fig, dpi=100):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    return img


def mae(im_true, im_pred):
    im_true, im_pred = np.array(im_true).reshape([-1]), np.array(im_pred).reshape([-1])
    return np.mean(np.abs(im_true - im_pred))


def pixel_metrics (im_true, im_pred, masked=False):
    #
    if not masked: 
        m = mae (np.squeeze(im_true), np.squeeze(im_pred))

        data_range = abs(max(im_true.max(), im_pred.max()) - min(im_true.min(), im_pred.min()))
        p = psnr(np.squeeze(im_true), np.squeeze(im_pred), data_range=data_range)
        s = ssim(np.squeeze(im_true), np.squeeze(im_pred), data_range=data_range)
    else: 
        real_range = abs(im_true.max() - im_true.min())
        true_mask = np.squeeze(im_true).copy() > im_true.min() + real_range * 0.1
        true_masked = np.squeeze(im_true)[true_mask]
        pred_masked = np.squeeze(im_pred)[true_mask]
        m = mae (np.squeeze(true_masked), np.squeeze(pred_masked))

        data_range = abs(max(true_masked.max(), pred_masked.max()) - min(true_masked.min(), pred_masked.min()))
        p = psnr(np.squeeze(true_masked), np.squeeze(pred_masked), data_range=data_range)
        s = ssim(np.squeeze(true_masked), np.squeeze(pred_masked), data_range=data_range)
    
    return m, s, p


def extract_roi_from_points (images, points, draw = False, roi_size = 64):
    #
    rois = []
    for k, image in enumerate(images): 
        #
        p_i = (points[0][0], points[0][1]); p_e = (points[1][0], points[1][1]); 
        
        if abs(p_i[0] - p_e[0]) <= 5: p_e = (points[1][0]+5, points[1][1]+5)
        if abs(p_i[1] - p_e[1]) <= 5: p_e = (points[1][0]+5, points[1][1]+5)

        # if abs(p_i[0] - p_e[0]) <= 10: p_e[0]+20
        #     if p_e[0] <= p_i[0]: p_e[0] = p_i[0]+20
        # if abs(p_i[1] - p_e[1]) <= 10: 
        #     if p_e[1] <= p_i[1]: p_e[1] = p_i[1]+20
        
        if draw == True: 
            image = cv2.rectangle(np.squeeze(image), p_i, p_e, color=0, thickness=1)
        
        image = np.squeeze(image)
        roi = image[min(p_i[1],p_e[1]) : max(p_i[1],p_e[1]), min(p_i[0],p_e[0]) : max(p_i[0],p_e[0])]
        
        roi = cv2.resize(roi, (roi_size, roi_size))
        rois.append(roi)
        
    return np.array(rois)


def extract_random_roi (ref_, in_, out_, fake_, draw = False, \
                        threshold = None, roi_size = 32, p_x=None, p_y=None):
    #
    if p_x == None or p_y == None:
        # Compute CE 
        real_ce = out_ - in_ #real_in - real_out #
        fake_ce = fake_ - in_ #real_in - fake_out #
        # ce_maps = ce_loss(fake_ce, real_ce)
        
        if threshold == None: 
            th = abs(torch.amin(real_ce, axis=(1,2,3))-torch.amax(real_ce, axis=(1,2,3)))*0.1
            mask_i, mask_s = torch.zeros_like(real_ce), torch.zeros_like(real_ce)
            for j, (s_ca, sth) in enumerate(zip(real_ce, th)): 
                mask_i[j] = s_ca < -sth
                mask_s[j] = s_ca >  sth
        else:
            #th = 0.5
            mask_i, mask_s = real_ce < -threshold, real_ce >  threshold

        mask_ = (mask_i + mask_s).to(torch.bool) # * 1.0
        # mask_r_flat = real_ce[mask_]
        # mask_f_flat = fake_ce[mask_]

        x,y,it = 0,0,0
        while mask_.squeeze()[x,y] == False or \
            (x + roi_size/2 >= mask_.shape[-2] or x - roi_size/2 < 0) or \
            (y + roi_size/2 >= mask_.shape[-1] or y - roi_size/2 < 0):
            x = np.random.randint(0, mask_.shape[-2])
            y = np.random.randint(0, mask_.shape[-1])
            it += 1
            if it > 1000:
                x,y = 128,128 
                break
        
        p_x, p_y = (x - int(roi_size/2), x + int(roi_size/2)), (y - int(roi_size/2), y + int(roi_size/2))
    
    # image = np.squeeze(image)
    roi_ref  = ref_.squeeze() [p_x[0]: p_x[1], p_y[0]: p_y[1]]
    roi_in   = in_.squeeze()  [p_x[0]: p_x[1], p_y[0]: p_y[1]]
    roi_out  = out_.squeeze() [p_x[0]: p_x[1], p_y[0]: p_y[1]]
    roi_fake = fake_.squeeze()[p_x[0]: p_x[1], p_y[0]: p_y[1]]

    if draw == True: 
        _, axes = plt.subplots(2,5,figsize=(20,8))
        axes = axes.ravel()
        axes[0].imshow(draw_roi_tensor(ref_, p_x, p_y), cmap='gray')
        axes[1].imshow(draw_roi_tensor(in_,  p_x, p_y), cmap='gray')
        axes[2].imshow(draw_roi_tensor(out_, p_x, p_y), cmap='gray')
        axes[3].imshow(draw_roi_tensor(fake_,p_x, p_y), cmap='gray')

        axes[5].imshow(roi_ref.squeeze().detach().cpu().numpy(), cmap='gray')
        axes[6].imshow(roi_in.squeeze().detach().cpu().numpy(),  cmap='gray')
        axes[7].imshow(roi_out.squeeze().detach().cpu().numpy(), cmap='gray')
        axes[8].imshow(roi_fake.squeeze().detach().cpu().numpy(),cmap='gray')

        for ax in axes: ax.set_axis_off()
        plt.tight_layout()
        # plt.savefig(save_image, dpi=200) #, bbox_inches='tight', pad_inches=0
        return roi_ref, roi_in, roi_out, roi_fake, axes
    else:
        return roi_ref, roi_in, roi_out, roi_fake


def extract_random_roi_points (in_, out_, threshold = None, roi_size = 32):
    #
    # Compute CE 
    real_ce = out_ - in_ #real_in - real_out #
    
    if threshold == None: 
        #####################################
        # th = abs(torch.amin(real_ce, axis=(1,2,3))-torch.amax(real_ce, axis=(1,2,3)))*0.1
        # mask_i, mask_s = torch.zeros_like(real_ce), torch.zeros_like(real_ce)
        # for j, (s_ca, sth) in enumerate(zip(real_ce, th)): 
        #     mask_i[j] = s_ca < -sth
        #     mask_s[j] = s_ca >  sth
        #####################################
        
        mask_i, mask_s = torch.zeros_like(real_ce), torch.zeros_like(real_ce)
        for j, s_ca in enumerate(real_ce): 
            # sth = abs(filters.threshold_otsu(s_ca.detach().cpu().numpy()))
            smooth = filters.gaussian(s_ca.detach().cpu().numpy(), sigma=2.5)
            sth = abs(filters.threshold_otsu(smooth))
            mask_i[j] = s_ca < -sth
            mask_s[j] = s_ca >  sth
    else:
        #th = 0.5
        mask_i, mask_s = real_ce < -threshold, real_ce >  threshold

    mask_ = (mask_i + mask_s).to(torch.bool) # * 1.0
    # mask_r_flat = real_ce[mask_]
    # mask_f_flat = fake_ce[mask_]

    x,y,it = 0,0,0
    while mask_.squeeze()[x,y] == False or \
          (x + roi_size/2 >= mask_.shape[-2] or x - roi_size/2 < 0) or \
          (y + roi_size/2 >= mask_.shape[-1] or y - roi_size/2 < 0):
        x = np.random.randint(0, mask_.shape[-2])
        y = np.random.randint(0, mask_.shape[-1])
        it += 1
        if it > 1000:
            x,y = 128,128 
            break
    
    p_x, p_y = (x - int(roi_size/2), x + int(roi_size/2)), (y - int(roi_size/2), y + int(roi_size/2))
    return p_x, p_y


def draw_roi_tensor (im, px, py):
    #
    assert isinstance(im, torch.Tensor) or \
           isinstance(im, torch.cuda.Tensor), \
           ValueError("Not a torch.Tensor or torch.cuda.Tensor")
    c = np.max(im.squeeze().detach().cpu().numpy())
    image = cv2.rectangle(im.squeeze().detach().cpu().numpy(), [min(py), min(px)], [max(py), max(px)], color=int(c), thickness=1)
    # ax[0].imshow(image, cmap='gray')
    return image


def estimate_confusion (real, prediction, name, save_figs = True, to_return = "line"):
    ## Confusion matrix 
    i_ranges = np.arange(0,101,10)
    real_ranges = []; real_lims = [];
    pred_ranges = []; pred_lims = [];
    allowed_labels = [];
    segm_real = np.zeros(np.squeeze(real).shape) * (-1)
    segm_prediction = np.zeros(np.squeeze(prediction).shape) * (-1)
    n = 0
    
    for k, (inf, sup) in enumerate(zip(i_ranges[:-1], i_ranges[1:])):
        #print (sup, inf)
        inf_val, sup_val = np.percentile(real, inf), np.percentile(real, sup)
        
        if round(inf_val, 6) != round(sup_val, 6) or round(inf_val, 6) == round(sup_val, 6) : 
            real_ranges.append([inf, sup])
            real_lims.append([inf_val, sup_val])
        else:
            if k != 0: 
                try: 
                    real_ranges[0] = [i_ranges[0], sup]
                    real_lims[0] = [np.percentile(real, i_ranges[0]), sup_val]
                except: 
                    real_ranges.append([i_ranges[0], sup])
                    real_lims.append([np.percentile(real, i_ranges[0]), sup_val])
    
    n = 0
    for k, (inf, sup) in enumerate(real_ranges):
    #for k, (inf, sup) in enumerate(zip(i_ranges[:-1], i_ranges[1:])):
        inf_pred, sup_pred = np.percentile(prediction, inf), np.percentile(prediction, sup)
        pred_lims.append([inf_pred, sup_pred])
        inf_area_pred = (inf_pred < np.squeeze(prediction))*1
        sup_area_pred = (np.squeeze(prediction) <= sup_pred)*1
        full_prediction = ((inf_area_pred + sup_area_pred)>1)*1
        segm_prediction = segm_prediction + full_prediction*(n); #print (n)
        #n += 1
        
        inf_real, sup_real = np.percentile(real, inf), np.percentile(real, sup)
        inf_area_real = (inf_real < np.squeeze(real))*1
        sup_area_real = (np.squeeze(real) <= sup_real)*1
        
        full_real = ((inf_area_real + sup_area_real)>1)*1
        segm_real = segm_real + full_real*(n); #print (n)
        allowed_labels.append(n)
        n += 1
    
    conf_real = confusion_matrix(segm_real.flatten(), segm_real.flatten(), labels = allowed_labels)
    conf_prediction = confusion_matrix(segm_real.flatten(), segm_prediction.flatten(), labels = allowed_labels)
    
    if conf_prediction.shape[0]%2 != 0:
        rh1 = np.linspace(0.5, 1, np.int(conf_prediction.shape[0]/2)+1); rh1 = np.roll(rh1, 1)
        rh2 = np.linspace(1, 1.5, np.int(conf_prediction.shape[0]/2)+1)
        r1 = np.concatenate([rh1, rh2[1:]])
    else: 
        rh1 = np.linspace(0.5, 1, np.int(conf_prediction.shape[0]/2)+1); rh1 = np.roll(rh1, 1)
        rh2 = np.linspace(1, 1.5, np.int(conf_prediction.shape[0]/2))
        r1 = np.concatenate([rh1, rh2[1:]])
    
    weights = []; weights.append(r1)
    for i in range(1, conf_prediction.shape[0]): weights.append(np.roll(r1, i))
    weights = np.asarray(weights)
    weights_tr = np.triu(weights, k=1)
    weights_f = np.zeros(weights.shape)
    weights_f = weights_tr + weights_tr.T; np.fill_diagonal(weights_f, weights.diagonal())
    
    mat = conf_prediction*weights_f
    FP = mat.sum(axis=0) - np.diag(mat)
    FN = mat.sum(axis=1) - np.diag(mat)
    TP = np.diag(mat)
    TN = mat.sum() - (FP + FN + TP)
    #print (FP, FN, TP, TN)
    
    Prec = np.nan_to_num(TP/(TP+FP)); Rec = np.nan_to_num(TP/(TP+FN)); 
    F1 = np.nan_to_num(2*Prec*Rec/(Prec + Rec));
    Acc = np.nan_to_num((TP+TN)/(TP+FP+FN+TN)); Spec = np.nan_to_num(TN/(TN+FP))
    #print (Prec, "\n", Rec, "\n", F1, "\n", Acc, "\n", Spec, "\n")
    
    pr = np.mean(Prec); rc = np.mean(Rec); fsc = np.mean(F1)
    acc = np.mean(Acc); sp = np.mean(Spec)
    #print (Prec, Rec, F1, Acc, Spec)
    
    line = "\nPrec = {0:4f} \nRec = {1:4f} \nF1 = {2:4f} \nAcc = {3:4f}".format(pr, rc, fsc, acc)
    #print (line)
    
    ticks_ranges = ["{0},{1}".format(e[0], e[1]) for e in real_ranges ]
    ticks_lims = ["{0:.2f}, {1:.4f}".format(e[0], e[1]) for e in real_lims ]
    #real_ticks = [x+"\n"+y for x,y in zip(ticks_ranges, ticks_lims)]
    ticks_lims = ["{0:.2f}, {1:.4f}".format(e[0], e[1]) for e in pred_lims ]
    #pred_ticks = [x+"\n"+y for x,y in zip(ticks_ranges, ticks_lims)]
    
    # # C / C.astype(np.float).sum(axis=1)
    # _, axes = plt.subplots(1,2, figsize=(16,5))
    # sns.heatmap(conf_prediction, annot=True, fmt=".2f", cmap="hot", xticklabels=real_ticks, yticklabels=pred_ticks, ax = axes[0]) 
    # sns.heatmap(np.divide(conf_prediction.T, conf_real.diagonal()).T, annot=True, fmt=".2f", cmap="hot", xticklabels=real_ticks, yticklabels=pred_ticks, ax = axes[1])
    # plt.tight_layout()
    
    _, axes = plt.subplots(1,2, figsize=(16,5))
    sns.heatmap(np.divide(conf_prediction.T, conf_real.diagonal()).T, annot=True, fmt=".2f", cmap="hot", vmin=0, vmax=1, ax = axes[0]) 
    sns.heatmap(np.divide(mat.T, conf_real.diagonal()).T, annot=True, fmt=".2f", cmap="hot", vmin=0, vmax=1, ax = axes[1]) #xticklabels=real_ticks, yticklabels=pred_ticks,
    plt.tight_layout()
    
    # Rotate the tick labels and set their alignment.
    #plt.setp(axes.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    
    if save_figs == True:
        fig = plt.gcf()
        fig = fig2img(fig)
        fig.save("{0}".format(name))
        plt.clf(); plt.close("all")
        
        # if os.path.isfile("{0}".format(name)):
        #     #plt.savefig("{0}".format(name) + str(np.random.randint(1,100)) + ".png")
        #     fig = plt.gcf()
        #     fig = fig2img(fig)
        #     fig.save("{0}".format(name) + str(np.random.randint(1,100)))
        #     plt.clf(); plt.close("all")
        # else: 
        #     #plt.savefig("{0}".format(name) + ".png")
        #     fig = plt.gcf()
        #     fig = fig2img(fig)
        #     fig.save("{0}".format(name))
        #     plt.clf(); plt.close("all")
    else: 
        plt.show()
    
    if to_return == "line":
        return line
    else:
        #return conf_real, np.divide(mat.T, conf_real.diagonal()).T, Prec, Rec, F1, Acc, Spec
        #return conf_real, conf_prediction, pr, rc, fsc, acc, sp
        return conf_real, np.nan_to_num(np.divide(mat.T, conf_real.diagonal()).T), pr, rc, fsc, acc, sp
        

def create_mask_otsu(f2r, dilated_mask = False, namefile="0.png", return_mask = True):
        #
        f2r_o = np.copy(f2r)
        f2r_eq = (f2r - np.min(f2r))/(np.max(f2r) - np.min(f2r))*255
        f2r_eq = np.uint8(f2r_eq)
        # Equalization
        f2r_eq = cv2.equalizeHist(f2r_eq)
        
        # Otsu
        blur = cv2.GaussianBlur(f2r_eq,(3,3),0)
        _, f2r_eq = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        f2r = np.copy(f2r_eq)
        
        dilated = dilation(np.squeeze(f2r), square(5)) / 255.0
        roi_ff2 = f2r_o.copy() # f2r.copy()
        if dilated_mask == True: roi_ff2 = np.squeeze(roi_ff2) * np.squeeze(dilated) 
        #else: roi_ff2 = np.squeeze(roi_ff2) * (np.squeeze(roi_ord)/255.0)
        else: roi_ff2 = np.squeeze(roi_ff2) #* (np.squeeze(roi_ord))
        
        f2_maxima = roi_ff2[np.where(roi_ff2!=0)]
        #print (np.min(roi_ff2), np.max(roi_ff2))
        
        if return_mask == False:
            f, fig = plt.subplots(1,3, figsize=(25,25))
            
            fig[0].imshow(np.squeeze(f2r_o), cmap='gray');   fig[0].set_axis_off()
            fig[1].imshow(np.squeeze(dilated), cmap='gray'); fig[1].set_axis_off()
            fig[2].imshow(np.squeeze(roi_ff2), cmap='gray'); fig[2].set_axis_off()
            #plt.show()
            plt.tight_layout()
            #plt.savefig(namefile)
            fig = plt.gcf()
            fig = fig2img(fig)
            fig.save(namefile)
            plt.clf(); plt.close("all")
        else: 
            return dilated


def dice_between_masks (mask_r, mask_f, k): 
    #
    intersection = np.sum(mask_f[mask_r == k]) * 2.0
    dice = intersection / (np.sum(mask_f) + np.sum(mask_r))
    return dice


def IoU_between_masks (mask_r, mask_f, k, cuda = False): 
    #
    if cuda:
        import torch
        intersection = torch.sum(mask_f[mask_r == k])
        union = torch.sum(mask_f) + torch.sum(mask_r) - intersection
        IoU = intersection / union
    else: 
        intersection = np.sum(mask_f[mask_r == k])
        union = np.sum(mask_f) + np.sum(mask_r) - intersection
        IoU = intersection / union
    return IoU


def plot_masks(roi_r, roi_f, mask_r, mask_f, dice, iou, namefile="0.png"):
    #
    f, fig = plt.subplots(2,2, figsize=(25,25))
    fig = fig.ravel()
    
    fig[0].imshow(np.squeeze(roi_r), cmap='gray');  fig[0].set_axis_off()
    fig[1].imshow(np.squeeze(roi_f), cmap='gray');  fig[1].set_axis_off()
    fig[2].imshow(np.squeeze(mask_r), cmap='gray'); fig[2].set_axis_off()
    fig[3].imshow(np.squeeze(mask_f), cmap='gray'); fig[3].set_axis_off()
    fig[0].set_title("ROI_real" , fontsize=30); fig[1].set_title("ROI_synth" , fontsize=30)
    fig[2].set_title("mask_real", fontsize=30); fig[3].set_title("mask_synth", fontsize=30)
    f.suptitle("Dice = {0:.4f} - IoU = {1:.4f}".format(dice, iou), fontsize=30)
    plt.tight_layout()
    #plt.savefig(namefile)
    fig = plt.gcf()
    fig = fig2img(fig)
    fig.save(namefile)
    plt.clf(); plt.close("all")

