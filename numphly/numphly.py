import cv2
import itertools
import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.stats.api as sms
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import pearsonr, shapiro, levene, f_oneway, kruskal
import scikit_posthocs as sp
import statsmodels.stats.multicomp as mc


arena_parameters = {
    'well_radius': 16.8,  # mm
    'food_cup_radius': 3.5,  # mm
}
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
font = cv2.FONT_HERSHEY_SIMPLEX
rgb = (255, 255, 255)

def frames_to_TC (frames, frame_rate):
    h = int(frames / frame_rate / 60 /60) 
    m = int(frames / frame_rate / 60) % 60 
    s = int((frames % (frame_rate*60))/frame_rate) 
    f = frames % (frame_rate*60) % frame_rate
    
    return ( "%02d:%02d:%02d:%02d" % ( h, m, s, f))


def is_point_between(a, b, c):
    # Extract coordinates
    ax = a[0,:];ay = a[1,:]
    bx = b[0,:];by = b[1,:]
    cx = c[0,:];cy = c[1,:]

    # Calculate vector AB (from A to B)
    abx = bx - ax
    aby = by - ay

    # Calculate vector AC (from A to C)
    acx = cx - ax
    acy = cy - ay

    # Compute the dot product of AB and AC
    dot_product = abx * acx + aby * acy

    # Compute the squared length of AB
    length_sq_ab = abx**2 + aby**2
#     # Check if A and B are the same point
#     if length_sq_ab == 0:
#         return (cx == ax) and (cy == ay)

    # Calculate the projection parameter t
    t = dot_product / length_sq_ab
    tracker_off_center = np.logical_and(t <= 1, t >= 0)
    # Determine if the projection of C lies between A and B (inclusive)
    return tracker_off_center



def clip_video(video, output_folder, output_file_prefix, region, buffer,frame_rate, height, width, file_type, loc, d_xy, parameter_mm2p):

    cap = cv2.VideoCapture(video)
    if file_type == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    elif file_type == '.mp4':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    color_loc = [(0,0,255),(0,255,0),]
        
    identifier = output_file_prefix +'_'+format(region,'06d')+file_type
    output_filepath = output_folder + '/' + identifier

    out = cv2.VideoWriter(output_filepath, fourcc, frame_rate, (height,width))
    
    d_f1f2_m   = np.squeeze(np.sqrt(np.sum(np.square(np.diff(loc,axis=0)),axis=1)))*parameter_mm2p
    for j in range(buffer):
        cap.set(cv2.CAP_PROP_POS_FRAMES, region+j)
        res, frame = cap.read()
        for i_fly in range(2):
            loc_i = loc[:,:,j]
            dot_xy = (loc_i[i_fly,:]-d_xy).astype('int')
            try:
                cv2.circle(frame, dot_xy, i_fly+3, color_loc[i_fly], 5)

            except:
                xxxx = 1
        try:          
            cv2.putText(frame, 'distance: ' + "%.1f" %d_f1f2_m[j] + ' mm',
                    (5,80), font, .5, (255,255,255), 1)
        except:
            xxxx = 1
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_filepath


def clip_videos(video, output_folder, output_file_prefix, event, event_loc, buffer_0, buffer_1, frame_rate, height, width, label, file_type):
    video_clip_list = []
    cap = cv2.VideoCapture(video)
    frame_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if file_type == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    elif file_type == '.mp4':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    font = cv2.FONT_HERSHEY_SIMPLEX
    rgb = (255, 255, 255)
    event_ts_index = 0
    for region in event:
        x = event_loc[event_ts_index,0]
        y = event_loc[event_ts_index,1]
        
        identifier = output_file_prefix+'_'+format(event_ts_index,'04d')+'_'+format(region,'06d')+file_type
        output_filepath = output_folder + '/' + identifier

        out = cv2.VideoWriter(output_filepath, fourcc, frame_rate, (height,width))
        if region+buffer_1 < frame_n:
            buffer_X = buffer_1
        else:
            buffer_X = frame_n - region
            print('event close to the end of experiment:',buffer_X,region+buffer_X)
        for j in range(-buffer_0, buffer_X):
            cap.set(cv2.CAP_PROP_POS_FRAMES, region+j)
            res, frame = cap.read()
            if label:
                foo = cv2.putText(frame, format(region+j,'06d'), (5, 520), font, .9, rgb)
            npframe = np.asarray(frame)
            npframe = npframe[int(y-height/2):int(y+height/2),int(x-width/2):int(x+width/2)]
            out.write(npframe)
        video_clip_list.append(identifier)
        event_ts_index += 1
#         if int(x-width/2)<=0 or int(y-height/2)<=0 or int(y+height/2) >= 576 or int(x+width/2) >= 576:
#             print(npframe.shape, x, y, frames_to_TC (region, frame_rate))
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return video_clip_list
    

    
def clip_videos_name(video, output_folder, output_file_prefix, event, event_loc, buffer_0, buffer_1, frame_rate, height, width, label, file_type):
    video_clip_list = []
    cap = cv2.VideoCapture(video)
    if file_type == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    elif file_type == '.mp4':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    font = cv2.FONT_HERSHEY_SIMPLEX
    rgb = (255, 255, 255)
    event_ts_index = 0
    for region in event:
        x = event_loc[event_ts_index,0]
        y = event_loc[event_ts_index,1]
        
        identifier = output_file_prefix+'_'+format(event_ts_index,'04d')+'_'+format(region,'06d')+file_type
        video_clip_list.append(identifier)
        event_ts_index += 1

    return video_clip_list
    

def tracking_analysis_body_center(fly_tracker_temp):
    fly_tracker_temp_3parts = (np.logical_or(fly_tracker_temp==1,fly_tracker_temp==2))[:,0:3,:]

    fly_tracker_temp_ht = np.logical_and(fly_tracker_temp_3parts[:,0,:],
                                         fly_tracker_temp_3parts[:,2,:])

    fly_tracker_temp_c  =  fly_tracker_temp_3parts[:,1,:]
    fly_tracker_temp_cr = (fly_tracker_temp == 3)[:,3,:]
    ##################### 6 cases##########################
    # both flies h and t are tracked
    well_tracker_ht     = np.logical_and(fly_tracker_temp_ht[0,:],fly_tracker_temp_ht[1,:])
    #both flies c are tracked
    well_tracker_c      = np.logical_and(fly_tracker_temp_c[0,:], fly_tracker_temp_c[1,:])
    #both flies cross are tracked
    well_tracker_cr     = np.logical_and(fly_tracker_temp_cr[0,:],fly_tracker_temp_cr[1,:])
    
    # one is ht, the other is center
    well_tracker_ht0_c1 = np.logical_and(fly_tracker_temp_ht[0,:],fly_tracker_temp_c[1,:])
    well_tracker_ht1_c0 = np.logical_and(fly_tracker_temp_ht[1,:],fly_tracker_temp_c[0,:])
    # one is cross, the other is center
    well_tracker_c0_cr1  = np.logical_and(fly_tracker_temp_c[0,:], fly_tracker_temp_cr[1,:])
    well_tracker_c1_cr0  = np.logical_and(fly_tracker_temp_c[1,:], fly_tracker_temp_cr[0,:])
    # one is ht, the other is cross
    well_tracker_ht0_cr1 = np.logical_and(fly_tracker_temp_ht[0,:],fly_tracker_temp_cr[1,:])
    well_tracker_ht1_cr0 = np.logical_and(fly_tracker_temp_ht[1,:],fly_tracker_temp_cr[0,:])
    # ht and c
    well_tracker_ht_c_e = np.logical_or(well_tracker_ht,  well_tracker_c)
    well_tracker_ht_c_x = np.logical_or(well_tracker_ht0_c1,  well_tracker_ht1_c0)
    well_tracker_ht_c   = np.logical_or(well_tracker_ht_c_e,  well_tracker_ht_c_x)
    # c and cross 
    well_tracker_c_cr_e = np.logical_or(well_tracker_c,  well_tracker_cr)
    well_tracker_c_cr_x = np.logical_or(well_tracker_c0_cr1,  well_tracker_c1_cr0)
    well_tracker_c_cr   = np.logical_or(well_tracker_c_cr_e,  well_tracker_c_cr_x)
    # ht and cross
    well_tracker_ht_cr_e = np.logical_or(well_tracker_ht,  well_tracker_cr)
    well_tracker_ht_cr_x = np.logical_or(well_tracker_ht0_cr1,  well_tracker_ht1_cr0)
    well_tracker_ht_cr   = np.logical_or(well_tracker_ht_cr_e,  well_tracker_ht_cr_x)

    well_tracker       = np.logical_or(well_tracker_ht_c,  well_tracker_c_cr)
    well_tracker       = np.logical_or(well_tracker_ht_cr,  well_tracker)
    return well_tracker



def tracking_analysis_2partsOrCr(fly_tracker_temp):
    fly_tracker_temp_3parts = (np.logical_or(fly_tracker_temp==1,fly_tracker_temp==2))[:,0:3,:]

    fly_tracker_temp_ht = np.logical_and(fly_tracker_temp_3parts[:,0,:],
                                         fly_tracker_temp_3parts[:,2,:])
    fly_tracker_temp_ct = np.logical_and(fly_tracker_temp_3parts[:,1,:],
                                         fly_tracker_temp_3parts[:,2,:])
    fly_tracker_temp_hc = np.logical_and(fly_tracker_temp_3parts[:,0,:],
                                         fly_tracker_temp_3parts[:,1,:])
                
    fly_tracker_temp_cr = (fly_tracker_temp == 3)[:,3,:]
    ##################### 6 cases##########################
    # both flies ht are tracked
    well_tracker_ht     = np.logical_and(fly_tracker_temp_ht[0,:],fly_tracker_temp_ht[1,:])
    #both flies ct are tracked
    well_tracker_ct      = np.logical_and(fly_tracker_temp_ct[0,:], fly_tracker_temp_ct[1,:])
    #both flies hc are tracked
    well_tracker_hc     = np.logical_and(fly_tracker_temp_hc[0,:],fly_tracker_temp_hc[1,:])
    #both flies cr are tracked
    well_tracker_cr     = np.logical_and(fly_tracker_temp_cr[0,:],fly_tracker_temp_cr[1,:])

    # one is ht, the other is ct
    well_tracker_ht0_ct1 = np.logical_and(fly_tracker_temp_ht[0,:],fly_tracker_temp_ct[1,:])
    well_tracker_ht1_ct0 = np.logical_and(fly_tracker_temp_ht[1,:],fly_tracker_temp_ct[0,:])
    # one is ct, the other is hc
    well_tracker_ct0_hc1 = np.logical_and(fly_tracker_temp_ct[0,:], fly_tracker_temp_hc[1,:])
    well_tracker_ct1_hc0 = np.logical_and(fly_tracker_temp_ct[1,:], fly_tracker_temp_hc[0,:])
    # one is ht, the other is hc
    well_tracker_ht0_hc1 = np.logical_and(fly_tracker_temp_ht[0,:],fly_tracker_temp_hc[1,:])
    well_tracker_ht1_hc0 = np.logical_and(fly_tracker_temp_ht[1,:],fly_tracker_temp_hc[0,:])

    # one is cr, the other is hc/ct/ht
    well_tracker_cr0_ht1 = np.logical_and(fly_tracker_temp_cr[0,:], fly_tracker_temp_ht[1,:])
    well_tracker_cr1_ht0 = np.logical_and(fly_tracker_temp_cr[1,:], fly_tracker_temp_ht[0,:])
    well_tracker_cr0_hc1 = np.logical_and(fly_tracker_temp_cr[0,:], fly_tracker_temp_hc[1,:])
    well_tracker_cr1_hc0 = np.logical_and(fly_tracker_temp_cr[1,:], fly_tracker_temp_hc[0,:])
    well_tracker_cr0_ct1 = np.logical_and(fly_tracker_temp_cr[0,:], fly_tracker_temp_ct[1,:])
    well_tracker_cr1_ct0 = np.logical_and(fly_tracker_temp_cr[1,:], fly_tracker_temp_ct[0,:])

    well_tracker_cr_ht_x = np.logical_or(well_tracker_cr0_ht1, well_tracker_cr1_ht0)
    well_tracker_cr_hc_x = np.logical_or(well_tracker_cr0_hc1, well_tracker_cr1_hc0)
    well_tracker_cr_ct_x = np.logical_or(well_tracker_cr0_ct1, well_tracker_cr1_ct0)

    well_tracker_cr_x = np.logical_or(well_tracker_cr_ht_x, well_tracker_cr_hc_x)
    well_tracker_cr_x = np.logical_or(well_tracker_cr_ct_x, well_tracker_cr_x)


    well_tracker_cr_ht_ct_hc = np.logical_or(well_tracker_cr_x, well_tracker_cr)
    # ht and ct
    well_tracker_ht_ct_e = np.logical_or(well_tracker_ht,  well_tracker_ct)
    well_tracker_ht_ct_x = np.logical_or(well_tracker_ht0_ct1, well_tracker_ht1_ct0)
    well_tracker_ht_ct   = np.logical_or(well_tracker_ht_ct_e, well_tracker_ht_ct_x)
    # ct and hc 
    well_tracker_ct_hc_e = np.logical_or(well_tracker_ct,  well_tracker_hc)
    well_tracker_ct_hc_x = np.logical_or(well_tracker_ct0_hc1,  well_tracker_ct1_hc0)
    well_tracker_ct_hc   = np.logical_or(well_tracker_ct_hc_e,  well_tracker_ct_hc_x)
    # ht and hc
    well_tracker_ht_hc_e = np.logical_or(well_tracker_ht,  well_tracker_hc)
    well_tracker_ht_hc_x = np.logical_or(well_tracker_ht0_hc1,  well_tracker_ht1_hc0)
    well_tracker_ht_hc   = np.logical_or(well_tracker_ht_hc_e,  well_tracker_ht_hc_x)

    well_tracker       = np.logical_or(well_tracker_ht_ct,  well_tracker_ct_hc)
    well_tracker       = np.logical_or(well_tracker_ht_hc,  well_tracker)
    well_tracker       = np.logical_or(well_tracker_cr_ht_ct_hc,  well_tracker)
    return well_tracker


def tracking_analysis_2partsOrCr_loc(loc):
    fly_tracker_temp_f1 = loc[0,:,0,:]>=0
    fly_tracker_temp_f2 = loc[1,:,0,:]>=0
    well_tracker_f1f2_temp_bp = np.logical_and(fly_tracker_temp_f1, fly_tracker_temp_f2)
    fly_tracker_f1_temp_2bp    = np.sum(fly_tracker_temp_f1[0:3,:], axis=0)>=2
    fly_tracker_f2_temp_2bp    = np.sum(fly_tracker_temp_f2[0:3,:], axis=0)>=2
    well_tracker_f1f2_temp_2bp = np.sum(well_tracker_f1f2_temp_bp[0:3,:],axis=0)>=2

    well_tracker_f1f2_temp_2bpcr = np.logical_and(fly_tracker_f1_temp_2bp,
                                                  fly_tracker_temp_f2[3,:])
    well_tracker_f1f2_temp_cr2bp = np.logical_and(fly_tracker_f2_temp_2bp,
                                                  fly_tracker_temp_f1[3,:])
    well_tracker_f1f2_temp = np.logical_or(well_tracker_f1f2_temp_cr2bp,
                                           well_tracker_f1f2_temp_2bpcr)
    well_tracker           = np.logical_or(well_tracker_f1f2_temp,
                                           well_tracker_f1f2_temp_2bp)
    return well_tracker
def tracking_analysis_2parts(fly_tracker_temp):
    fly_tracker_temp_3parts = (np.logical_or(fly_tracker_temp==1,fly_tracker_temp==2))[:,0:3,:]

    fly_tracker_temp_ht = np.logical_and(fly_tracker_temp_3parts[:,0,:],
                                         fly_tracker_temp_3parts[:,2,:])
    fly_tracker_temp_ct = np.logical_and(fly_tracker_temp_3parts[:,1,:],
                                         fly_tracker_temp_3parts[:,2,:])
    fly_tracker_temp_hc = np.logical_and(fly_tracker_temp_3parts[:,0,:],
                                         fly_tracker_temp_3parts[:,1,:])

    ##################### 6 cases##########################
    # both flies ht are tracked
    well_tracker_ht     = np.logical_and(fly_tracker_temp_ht[0,:],fly_tracker_temp_ht[1,:])
    #both flies ct are tracked
    well_tracker_ct      = np.logical_and(fly_tracker_temp_ct[0,:], fly_tracker_temp_ct[1,:])
    #both flies hc are tracked
    well_tracker_hc     = np.logical_and(fly_tracker_temp_hc[0,:],fly_tracker_temp_hc[1,:])

    well_tracker       = np.logical_or(well_tracker_ht,  well_tracker_hc)
    well_tracker       = np.logical_or(well_tracker_hc,  well_tracker)
    return well_tracker
def tracking_analysis_or(loc):
    n_ID = loc.shape[0]
    numBP = loc.shape[1]
    EXP_LENGTH = loc.shape[-1]
    fly_tracker = np.zeros((2,EXP_LENGTH))>0
    for fly_i in range(n_ID):
        for bpindex in range(numBP):
            x     = loc[fly_i,bpindex,0,:]
            y     = loc[fly_i,bpindex,1,:]
            Index = np.logical_and(~np.isnan(x), ~np.isnan(y))
            fly_tracker[fly_i,:] = np.logical_or(fly_tracker[fly_i,:], Index)
    return fly_tracker 


########
def tracking_analysis_or_cf(loc,conf,thr_cf):
    n_ID = loc.shape[0]
    numBP = loc.shape[1]
    frame_n = loc.shape[-1]
    fly_tracker_f_goodcf = np.ones((n_ID,numBP,frame_n))<0
    loc_temp          = np.zeros((n_ID,1,2,frame_n))
    loc_temp[:,:,:,:] = np.NaN
    for i_bp in range(numBP):
        loc_temp[:,0,:,:] = loc[:,i_bp,:,:]
        fly_tracker_f_bp  = tracking_analysis_or(loc_temp)
        for i_f in range(n_ID):
            fly_tracker_cf_if_bp = np.logical_or( 
                conf[i_f,i_bp,:]>=thr_cf,
                np.isnan(conf[i_f,i_bp,:]))
            fly_tracker_f_goodcf[i_f,i_bp,:] = np.logical_or(
                fly_tracker_f_goodcf[i_f,i_bp,:],
                np.logical_and(fly_tracker_f_bp[i_f,:], fly_tracker_cf_if_bp))

#     fly_tracker_f_goodcf_n = np.zeros((2,3,2,frame_n))==1
#     fly_tracker_f_goodcf_n[:,:,0,:] = ~fly_tracker_f_goodcf
#     fly_tracker_f_goodcf_n[:,:,1,:] = ~fly_tracker_f_goodcf
#     loc[fly_tracker_f_goodcf_n] = np.NaN
    return fly_tracker_f_goodcf

def tracking_analysis_and(loc):
    n_ID = loc.shape[0]
    numBP = loc.shape[1]
    EXP_LENGTH = loc.shape[-1]
    fly_tracker = np.zeros((2,EXP_LENGTH))==0
    for fly_i in range(n_ID):
        for bpindex in range(numBP):
            x     = loc[fly_i,bpindex,0,:]
            y     = loc[fly_i,bpindex,1,:]
            Index = np.logical_and(~np.isnan(x), ~np.isnan(y))
            fly_tracker[fly_i,:] = np.logical_and(fly_tracker[fly_i,:], Index)
    return fly_tracker


# def tracking_analysis_or(loc, conf, thr):
#     n_ID = loc.shape[0]
#     numBP = loc.shape[1]
#     EXP_LENGTH = loc.shape[-1]
#     fly_tracker = np.zeros((2,EXP_LENGTH))>0
#     for fly_i in range(n_ID):
#         for bpindex in range(numBP):
#             x     = loc[fly_i,bpindex,0,:]
#             y     = loc[fly_i,bpindex,1,:]
#             Index = np.logical_and(~np.isnan(x), ~np.isnan(y))
            
#             fly_tracker_cf = np.logical_or(conf[fly_i,bpindex,:]>=thr, np.isnan(conf[fly_i,bpindex,:]))
#             Index = np.logical_and(Index, fly_tracker_cf)
            
#             fly_tracker[fly_i,:] = np.logical_or(fly_tracker[fly_i,:], Index)
#     return fly_tracker 
    
# def tracking_analysis_and(loc, conf, thr):
#     n_ID = loc.shape[0]
#     numBP = loc.shape[1]
#     EXP_LENGTH = loc.shape[-1]
#     fly_tracker = np.zeros((2,EXP_LENGTH))==0
#     for fly_i in range(n_ID):
#         for bpindex in range(numBP):
#             x     = loc[fly_i,bpindex,0,:]
#             y     = loc[fly_i,bpindex,1,:]
#             Index = np.logical_and(~np.isnan(x), ~np.isnan(y))
#             fly_tracker_cf = np.logical_or(conf[fly_i,bpindex,:]>=thr, np.isnan(conf[fly_i,bpindex,:]))
#             Index = np.logical_and(Index, fly_tracker_cf)
#             fly_tracker[fly_i,:] = np.logical_and(fly_tracker[fly_i,:], Index)
#     return fly_tracker

def tracking_analysis_eql(loc, thr, thr_bp):
    n_ID = loc.shape[0]
    numBP = loc.shape[1]
    EXP_LENGTH = loc.shape[-1]
    fly_tracker_eql = np.zeros((EXP_LENGTH)) == 0
    idx_loc = (loc[0,:,:,:] == loc[1,:,:,:])
    d_c_f1f2 = np.sqrt(np.sum(np.square(loc[0,:,:,:]-loc[1,:,:,:]),axis=1))
    idx_loc = np.sum(d_c_f1f2 <= thr,axis=0)

    for i_f in range(EXP_LENGTH):
        if idx_loc[i_f] >= thr_bp:
            fly_tracker_eql[i_f] = False
    return fly_tracker_eql
    

def tracking_analysis_bp_outliers(loc, thres):
    
    n_ID = loc.shape[0]
    EXP_LENGTH = loc.shape[-1]
    
    d_f_p1p2 = np.sqrt(np.sum(np.square(loc[:,0,:,:]-loc[:,1,:,:]),axis=1))
    d_f_p1p3 = np.sqrt(np.sum(np.square(loc[:,0,:,:]-loc[:,2,:,:]),axis=1))
    d_f_p2p3 = np.sqrt(np.sum(np.square(loc[:,1,:,:]-loc[:,2,:,:]),axis=1))
    
    d_f_p1p2_m = np.nanmean(d_f_p1p2,axis=1)
    d_f_p1p3_m = np.nanmean(d_f_p1p3,axis=1)
    d_f_p2p3_m = np.nanmean(d_f_p2p3,axis=1)
    
    d_f_p1p2_std = np.nanstd(d_f_p1p2,axis=1)
    d_f_p1p3_std = np.nanstd(d_f_p1p3,axis=1)
    d_f_p2p3_std = np.nanstd(d_f_p2p3,axis=1)

    f_p1p2_tracker = np.zeros((2,EXP_LENGTH))==0
    f_p2p3_tracker = np.zeros((2,EXP_LENGTH))==0
    f_p1p3_tracker = np.zeros((2,EXP_LENGTH))==0

    for i in range(n_ID):
        f_p1p2_tracker[i,:] = np.logical_and(
            ~(d_f_p1p2[i,:]>=(d_f_p1p2_m[i]+d_f_p1p2_std[i]*thres)),
            ~(d_f_p1p2[i,:]<=(d_f_p1p2_m[i]-d_f_p1p2_std[i]*thres)))
        f_p2p3_tracker[i,:] = np.logical_and(
            ~(d_f_p2p3[i,:]>=(d_f_p2p3_m[i]+d_f_p2p3_std[i]*thres)),
            ~(d_f_p2p3[i,:]<=(d_f_p2p3_m[i]-d_f_p2p3_std[i]*thres)))
        f_p1p3_tracker[i,:] = np.logical_and(
            ~(d_f_p1p3[i,:]>=(d_f_p1p3_m[i]+d_f_p1p3_std[i]*thres)),
            ~(d_f_p1p3[i,:]<=(d_f_p1p3_m[i]-d_f_p1p3_std[i]*thres)))
        
    return f_p1p2_tracker,f_p2p3_tracker,f_p1p3_tracker
        
        
    
def read_loc(Dataframe,scorer,animalID, bodyparts):

    frames = Dataframe[scorer][animalID[0]][bodyparts[0]]['likelihood'].values
    frame_n = len(frames)
    n_ID = len(animalID)
    numBP = len(bodyparts)
    EXP_LENGTH = 126000
#     print('tracking len',frame_n)
    loc      = np.zeros((2,numBP,2,EXP_LENGTH))
    conf_loc = np.zeros((2,numBP,EXP_LENGTH))
    if frame_n<EXP_LENGTH:
        loc[:,:,:,frame_n:EXP_LENGTH] = np.NaN
        conf_loc[:,:,frame_n:EXP_LENGTH] = np.NaN
    fly_i = 0
    for fly in animalID:

        for bpindex, bp in enumerate(bodyparts):
            
            conf = Dataframe[scorer][fly][bp]['likelihood'].values
            conf_loc[fly_i,bpindex,0:frame_n] = conf
            x     = Dataframe[scorer][fly][bp]['x'].values
            y     = Dataframe[scorer][fly][bp]['y'].values
            loc[fly_i,bpindex,0,0:frame_n] = x
            loc[fly_i,bpindex,1,0:frame_n] = y
        fly_i += 1
    return loc, conf_loc

def read_loc_paint(Dataframe,scorer,uniqueID, paint):

    
    conf = Dataframe[scorer][uniqueID][paint]['likelihood'].values
    frame_n = len(conf)
    EXP_LENGTH = 126000
    loc      = np.zeros((2,EXP_LENGTH))
    conf_loc = np.zeros(EXP_LENGTH)
    if frame_n<EXP_LENGTH:
        loc[:,frame_n:EXP_LENGTH] = np.NaN
        conf_loc[frame_n:EXP_LENGTH] = np.NaN

    conf_loc[0:frame_n] = conf
    x     = Dataframe[scorer][uniqueID][paint]['x'].values
    y     = Dataframe[scorer][uniqueID][paint]['y'].values
    loc[0,0:frame_n] = x
    loc[1,0:frame_n] = y

    return loc, conf_loc


def filter_loc_ht(loc, fly_tracker, thr):             
    frame_n = loc.shape[-1]
    n_ID    = loc.shape[0]
    fly_tracker_and  = tracking_analysis_and(loc)
#     well_tracker_eql = tracking_analysis_eql(loc, 1, 1)
    fly_tracker_ht_off = np.zeros((n_ID,frame_n))>0
    for i_fly in range(n_ID):
        loc_h = loc[i_fly,0,:,:]
        loc_c = loc[i_fly,1,:,:]
        loc_t = loc[i_fly,2,:,:]
        d_hc = np.sqrt(np.sum(np.square(loc_h-loc_c),axis=0))
        d_ht = np.sqrt(np.sum(np.square(loc_h-loc_t),axis=0))
        d_ct = np.sqrt(np.sum(np.square(loc_c-loc_t),axis=0))
        tracker_ht_close = d_ht<thr
        tracker_hc_close = d_hc<thr
        tracker_ct_close = d_ct<thr
        tracker_bp_close = np.logical_or(np.logical_or(tracker_ht_close,tracker_hc_close),tracker_ct_close)

        tracker_off_center = ~is_point_between(loc_h, loc_t, loc_c)
        tracker_off_center = np.logical_and(tracker_off_center, fly_tracker_and[i_fly,:])
        fly_tracker_ht_off[i_fly,:] = np.logical_and(tracker_off_center,tracker_bp_close)

        loc[i_fly,0,:,fly_tracker_ht_off[i_fly,:]] = np.NaN
        loc[i_fly,2,:,fly_tracker_ht_off[i_fly,:]] = np.NaN

        fly_tracker[i_fly,0,fly_tracker_ht_off[i_fly,:]] = False
        fly_tracker[i_fly,2,fly_tracker_ht_off[i_fly,:]] = False
        
    return loc, fly_tracker, fly_tracker_ht_off
                

def filter_loc_HBorL(loc, well_tracker, thr_TT, d_d_threshold, thr_HC, buffer):
    Head1   = np.squeeze(loc[0,0,:,:])
    Center1 = np.squeeze(loc[0,1,:,:])
    tail1   = np.squeeze(loc[0,2,:,:])
    Head2   = np.squeeze(loc[1,0,:,:])
    Center2 = np.squeeze(loc[1,1,:,:])
    tail2   = np.squeeze(loc[1,2,:,:])
    d_T1T2 = np.sqrt(np.sum(np.square(tail1-tail2),axis=0))
    d_H1C2 = np.sqrt(np.sum(np.square(Head1-Center2),axis=0))
    d_H2C1 = np.sqrt(np.sum(np.square(Head2-Center1),axis=0))
    d_H1T2 = np.sqrt(np.sum(np.square(Head1-tail2),axis=0))
    d_H2T1 = np.sqrt(np.sum(np.square(Head2-tail1),axis=0))
    d_H1H2 = np.sqrt(np.sum(np.square(Head1-Head2),axis=0))
    
    dis_tracker_HC = np.logical_or(d_H1C2<thr_HC, d_H2C1<thr_HC)
    dis_tracker_HT = np.logical_or(d_H1T2<thr_HC, d_H2T1<thr_HC)
    dis_tracker = np.logical_or(dis_tracker_HC, dis_tracker_HT)
    dis_tracker = np.logical_or(dis_tracker, d_H1H2<thr_HC)
    dis_tracker = np.logical_and(dis_tracker, d_T1T2>thr_TT)
    
    # same body part distance between 2 flies
    d_loc = np.sqrt(np.sum(np.square(np.diff(loc,axis=0)),axis=2))
    # same body part distance change between 2 frames
    d_d_loc = np.diff(np.squeeze(d_loc),axis=1)


    dd_tracker = np.logical_or(d_d_loc[1,:]<=-d_d_threshold, d_d_loc[1,:]<=-d_d_threshold)
    dd_tracker = np.logical_or(d_d_loc[2,:]<=-d_d_threshold, dd_tracker)
    dd_tracker = np.logical_and(dis_tracker[:-1], dd_tracker)


    # frames where flies are close enough
    idx_frame_dd = (idx_frame[:-1])[dd_tracker]
    # frames where flies are tracked
    idx_frame_tracked = idx_frame[well_tracker]
    

#     NaN_gaps      = np.diff(idx_frame_tracked)
#     idx_frame_gap =(idx_frame_tracked[:-1])[np.logical_and(NaN_gaps>=2, NaN_gaps<=3)]
#     idx_frame_ms = np.zeros(frame_n)>0
#     for k in idx_frame_gap:
#         idx_frame_ms[(k-buffer):(k+buffer)] = True

    idx_frame_ms = np.zeros(frame_n)>0
    for k in idx_frame_dd:
        idx_frame_ms[(k-buffer):(k+buffer)] = True
        
    return idx_frame_ms

def tracker_ID_View_classifer(conf_1_i, conf_2_i, loc_1_m_i, loc_2_m_i, n_mismatch):
    
    loc_code = np.zeros((2,2))
    
    d12_11 = np.sqrt(np.sum(np.square(loc_1_m_i[0,:]-loc_2_m_i[0,:])))
    d12_12 = np.sqrt(np.sum(np.square(loc_1_m_i[0,:]-loc_2_m_i[1,:])))
    d12_21 = np.sqrt(np.sum(np.square(loc_1_m_i[1,:]-loc_2_m_i[0,:])))
    d12_22 = np.sqrt(np.sum(np.square(loc_1_m_i[1,:]-loc_2_m_i[1,:])))
    
    d11_12 = np.sqrt(np.sum(np.square(loc_1_m_i[0,:]-loc_1_m_i[1,:])))
    d22_12 = np.sqrt(np.sum(np.square(loc_2_m_i[0,:]-loc_2_m_i[1,:])))
    
    if d12_11 <= d12_12:      # fly 1 and fly 2 in w tracker and fly 1 in f tracker is viable
        if d12_22 <= d12_21:
            n_mismatch[0] = n_mismatch[0]+1
            loc_code[0,0] = 0
            loc_code[1,0] = 1
            if conf_1_i[0] >= conf_2_i[0]:
                loc_code[0,1] = 0
            else:
                loc_code[0,1] = 1

            if conf_1_i[1] >= conf_2_i[1]:
                loc_code[1,1] = 0
            else:
                loc_code[1,1] = 1
            
        elif d12_22 > d12_21:
            
            if d11_12 <= d22_12:
                n_mismatch[1] = n_mismatch[1]+1
                # f tracker, fly 1 is more accurate than fly2
                if conf_1_i[0] >= conf_1_i[1]: 
                    loc_code[1,0] = 1 # 
                    loc_code[1,1] = 1 #
                
                    if conf_1_i[0] >= conf_2_i[0] and conf_1_i[0] >= conf_1_i[1]:
                        loc_code[0,0] = 0
                        loc_code[0,1] = 0
                    elif conf_1_i[0] <= conf_2_i[0] and conf_1_i[1] <= conf_2_i[0]:
                        loc_code[0,0] = 0
                        loc_code[0,1] = 1
                    else:
                        loc_code[0,0] = 1
                        loc_code[0,1] = 0
                # f tracker, fly 2 is more accurate than fly1        
                else:                    
                    loc_code[0,0] = 1 # 
                    loc_code[0,1] = 1 #
                
                    if conf_1_i[0] >= conf_2_i[0] and conf_1_i[0] >= conf_1_i[1]:
                        loc_code[1,0] = 0
                        loc_code[1,1] = 0
                    elif conf_1_i[0] <= conf_2_i[0] and conf_1_i[1] <= conf_2_i[0]:
                        loc_code[1,0] = 0
                        loc_code[1,1] = 1
                    else:
                        loc_code[1,0] = 1
                        loc_code[1,1] = 0

            else:# 
                n_mismatch[2] = n_mismatch[2]+1
#                 print('fly 1 and 2 in f tracker both are close to fly 1 in w tracker')
#                 print('but fly pair distance in ftracker is bigger than the distance in w tracker')
        
        else: #fly 2 in 2nd tracker is NaN
            n_mismatch[3] = n_mismatch[3]+1
            loc_code[0,0] = 0
            loc_code[1,0] = 1
            loc_code[1,1] = 1
            if conf_1_i[0]>=conf_2_i[0]:
                loc_code[0,1] = 0 # floor tracker
            else:
                loc_code[0,1] = 1 # wall tracker
                
            


    elif d12_11 > d12_12:
        if d12_22 >= d12_21:
            n_mismatch[4] = n_mismatch[4]+1
            if conf_1_i[0] >= conf_2_i[1]:
                loc_code[0,0] = 0 # fly 1
                loc_code[0,1] = 0 # floor tracker
            else:
                loc_code[0,0] = 1 # fly 2
                loc_code[0,1] = 1 # wall tracker

            if conf_1_i[1] >= conf_2_i[0]:
                loc_code[1,0] = 1 # fly 2
                loc_code[1,1] = 0 # floor tracker
            else:
                loc_code[1,0] = 0 # fly 1
                loc_code[1,1] = 1 # wall tracker
        elif d12_22 < d12_21:
            # f tracker, fly 1 is more accurate than fly2
            
            if d11_12 <= d22_12:
                n_mismatch[5] = n_mismatch[5]+1
                # f tracker, fly 1 is more accurate than fly2
                conf_1_f1overf2_i = conf_1_i[0] >= conf_1_i[1]
                if conf_1_f1overf2_i:
                    
                    loc_code[1,0] = 0 # 
                    loc_code[1,1] = 1 #
                
                    if conf_1_i[0] >= conf_2_i[1] and conf_1_f1overf2_i:
                        loc_code[0,0] = 0
                        loc_code[0,1] = 0
                    elif conf_1_i[0] <= conf_2_i[1] and conf_1_i[1] <= conf_2_i[1]:
                        loc_code[0,0] = 1
                        loc_code[0,1] = 1
                    else:
                        loc_code[0,0] = 1
                        loc_code[0,1] = 0
                # f tracker, fly 2 is more accurate than fly1        
                else:                    
                    loc_code[0,0] = 0 # 
                    loc_code[0,1] = 1 #
                
                    if conf_1_i[0] >= conf_2_i[1] and conf_1_f1overf2_i:
                        loc_code[1,0] = 0
                        loc_code[1,1] = 0
                    elif conf_1_i[0] <= conf_2_i[1] and conf_1_i[1] <= conf_2_i[1]:
                        loc_code[1,0] = 1
                        loc_code[1,1] = 1
                    else:
                        loc_code[1,0] = 1
                        loc_code[1,1] = 0
                        

            else:
                n_mismatch[6] = n_mismatch[6]+1
#                 print('fly 1 and 2 in f tracker both are close to fly 2 in w tracker')
#                 print('but fly pair distance in ftracker is bigger than the distance in w tracker')
        # fly 2 in f tracker is not viable
        else: 
            n_mismatch[7] = n_mismatch[7]+1
            loc_code[1,0] = 0
            loc_code[1,1] = 1
            if conf_1_i[0]>=conf_2_i[1]:
                loc_code[0,0] = 0
                loc_code[0,1] = 0 # floor tracker
            else:
                loc_code[0,0] = 1
                loc_code[0,1] = 1 # wall tracker
    elif d12_21 >= d12_22:
        if d12_11 <= d12_12:
            n_mismatch[8] = n_mismatch[8]+1
        elif d12_11 <= d12_12:
            if d11_12 >= d22_12:
                n_mismatch[9] = n_mismatch[9]+1
            else:
                n_mismatch[10] = n_mismatch[10]+1
        else:
            n_mismatch[11] = n_mismatch[11]+1
    elif d12_21 < d12_22:
        if d12_11 >= d12_12:
            n_mismatch[12] = n_mismatch[12]+1
        elif d12_11 < d12_12:
            if d11_12 >= d22_12:
                n_mismatch[13] = n_mismatch[13]+1
            else:
                n_mismatch[14] = n_mismatch[14]+1
        else:
            n_mismatch[15] = n_mismatch[15]+1
    elif d11_12 >= 0:
        n_mismatch[16] = n_mismatch[16]+1
    elif d22_12 >= 0:
        n_mismatch[17] = n_mismatch[17]+1
    else:
        n_mismatch[18] = n_mismatch[18]+1
    return loc_code, n_mismatch



# def downsample(loc,window):
#     #
#     size = loc.shape
#     n_points = int(size[-1]/window)
#     loc_new = np.zeros((size[0],size[1],size[2],n_points))
    
#     for i in range(n_points):
#         i_range = np.arange(i*window,(i+1)*window)
#         loc_new[:,:,:,i] = np.nanmean(loc[:,:,:,i_range],axis=3)
    
#     return loc_new




def downsample(loc,window,thr_presence):
    
    size = loc.shape
    n_points = int(size[-1]/window)
    n_ID = size[0]
    n_BP = size[1]
    loc_new = np.zeros((size[0],size[1],size[2],n_points))
        
    if window == 1:
        loc_new[:,:,:,:] = loc
    else:
        #Data Presence Threshold
        # thr_presence = 0.1
        for i in range(n_points):
            i_range = np.arange(i*window,(i+1)*window)

            for i_fly in range(n_ID):
                for i_bp in range(n_BP):

                    n_bp_presence = np.sum(~np.isnan(loc[i_fly,i_bp,0,i_range]))

                    if n_bp_presence >= thr_presence*window:
                        loc_new[i_fly,i_bp,:,i] = np.nanmean(loc[i_fly,i_bp,:,i_range].T,axis=1)
                    else:
                        loc_new[i_fly,i_bp,:,i] = np.NaN
#             win_h = int(window/2)
#             i_range_1 = np.arange(i*window      , i*window+win_h)
#             i_range_2 = np.arange(i*window+win_h,(i+1)*window)
#             for i_fly in range(n_ID):
#                 for i_bp in range(n_BP):

#                     n_bp_presence = np.sum(~np.isnan(loc[i_fly,i_bp,0,i_range]))
#                     n_bp_presence_1 = np.sum(~np.isnan(loc[i_fly,i_bp,0,i_range_1]))
#                     n_bp_presence_2 = np.sum(~np.isnan(loc[i_fly,i_bp,0,i_range_2]))
                    
#                     if n_bp_presence_1 >= thr_presence*window and n_bp_presence_2 >= thr_presence*window:
    return loc_new

def find_farthest_points(points):
    max_distance = -1
    best_pair = (0, 1)
    all_nan = False
    # Check all pairs of points
    for i, j in itertools.combinations(range(len(points)), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5  # Euclidean distance
        if distance > max_distance:
            max_distance = distance
            all_nan = True
            best_pair = (i, j)
    
    return best_pair, max_distance
    


def slow_period(loc,s_thr,gap_thr, d_thr):
    d_loc_df = np.sqrt(np.sum(np.square(np.diff(loc,axis=3)),axis=2))
    idx_frame = np.arange(EXP_LENGTH)

    idx_frame_ms = np.zeros((2,EXP_LENGTH))>1
    idx_fly_floor = np.zeros((2,EXP_LENGTH))>1
    
    Head1   = np.squeeze(loc[0,0,:,:])
    Head2   = np.squeeze(loc[1,0,:,:])
    Tail1   = np.squeeze(loc[0,1,:,:])
    Tail2   = np.squeeze(loc[1,1,:,:])
    d_H1H2   = np.sqrt(np.sum(np.square(Head1-Head2),axis=0))
    d_T1T2   = np.sqrt(np.sum(np.square(Tail1-Tail2),axis=0))
    idx_floor = np.logical_and(d_T1T2 >= d_thr, d_H1H2 >= d_thr)
    
    
    for fly_i in range(2):
        dd_tracker = np.logical_and(d_loc_df[fly_i,0,:]<=s_thr, d_loc_df[fly_i,1,:]<=s_thr)

        gap_tracker = np.zeros(EXP_LENGTH-1)
        gap_tracker[dd_tracker] =1

        dd_tracker_d = np.diff(gap_tracker)

        gap_s = (idx_frame[:-2])[dd_tracker_d ==  1]
        gap_e = (idx_frame[:-2])[dd_tracker_d == -1]

        if gap_s[0] > gap_e[0]:
            gap_s = np.hstack((-1,gap_s))
        if gap_s[-1] > gap_e[-1]:
            gap_e = np.hstack((gap_e,len(idx_frame[:-1])))

        seg_len = gap_e-gap_s

        seg_len_thr = seg_len[seg_len>=gap_thr]
        gap_e_thr = gap_e[seg_len>=gap_thr]
        gap_s_thr = gap_s[seg_len>=gap_thr]
        if seg_len_thr.shape[0]==0:

            print(well,ii,sex,fly_i,': no motionless period')

        i_gap = 0
        for k in gap_s_thr:
            idx_frame_ms[fly_i,(k):(k+seg_len_thr[i_gap])] = True
            i_gap = i_gap+1

        idx_frame_ms[fly_i,:] = np.logical_and(idx_frame_ms[fly_i,:],idx_floor)
    
    return idx_frame_ms


# def tracker_gaps(fly_tracker):
#     gaps =   []
#     gaps_l = []
#     for i in range(2):
#         arr = fly_tracker[i,:]
#         mask = ~fly_tracker[i,:]
#         padded_mask = np.concatenate([[False], mask, [False]])
#         diffs = np.diff(padded_mask.astype(int))
#         starts = np.where(diffs == 1)[0]
#         ends = np.where(diffs == -1)[0] - 1
#         gaps_l.append(ends - starts + 1)
#         gaps.append(starts)
#     return gaps, gaps_l



def tracked_segments(well_tracker_size):
    mask = well_tracker_size
    padded_mask = np.concatenate([[False], mask, [False]])
    diffs = np.diff(padded_mask.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    gaps_l = ends - starts + 1
    gaps   = starts
    return gaps_l, gaps

def binary_segments(binary_array, time_per_step):
    segments = []
    n = len(binary_array)
    current_state = binary_array[0]
    start_time = 0
    
    for i in range(1, n):
        if binary_array[i] != current_state:
            end_time = i * time_per_step
            segments.append((current_state, start_time, end_time))
            current_state = binary_array[i]
            start_time = end_time
    # Add the last segment
    segments.append((current_state, start_time, n * time_per_step))
    return segments




def three_group_stat(group1, group2, group3, alpha):
    # Check Normality (Shapiro-Wilk)
    normality_results = [
        stats.shapiro(group1).pvalue,
        stats.shapiro(group2).pvalue,
        stats.shapiro(group3).pvalue
    ]
    print("Normality p-values:", [f"{p:.4f}" for p in normality_results])

    # Check Homogeneity of Variances (Levene’s test)
    levene_p = stats.levene(group1, group2, group3).pvalue
    print(f"Levene's p-value: {levene_p:.4f}")
    
    # Decide between ANOVA or Kruskal-Wallis
    if all(p > alpha for p in normality_results) and levene_p > alpha:
        print("\nUsing ANOVA...")
        anova_result = stats.f_oneway(group1, group2, group3)
        print(f"ANOVA p-value: {anova_result.pvalue:.8f}")
        if anova_result.pvalue < alpha:
            # Post-hoc Tukey HSD
            tukey = pairwise_tukeyhsd(
                endog=np.concatenate([group1, group2, group3]),
                groups=np.array(['Group1']*len(group1) + ['Group2']*len(group2) + ['Group3']*len(group3)),
                alpha=alpha
            )
            print(tukey.summary())
    else:
        print("\nUsing Kruskal-Wallis...")
        kw_result = stats.kruskal(group1, group2, group3)
        print(f"Kruskal-Wallis p-value: {kw_result.pvalue:.8f}")
        if kw_result.pvalue < alpha:
    #         Post-hoc Dunn's test (requires scikit-posthocs)
            import scikit_posthocs as sp
            data = [group1, group2, group3]
            dunn_result = sp.posthoc_dunn(data, p_adjust='bonferroni')
            print("\nDunn's test (Bonferroni-adjusted p-values):")
            print(dunn_result)
            
def two_group_stat(group1, group2):

    shapiro_a = stats.shapiro(group1)
    shapiro_b = stats.shapiro(group2)
    print(f"Shapiro-Wilk p-value (Group A): {shapiro_a.pvalue:.8f}")
    print(f"Shapiro-Wilk p-value (Group B): {shapiro_b.pvalue:.8f}")

    # Check variance equality
    levene_test = stats.levene(group1, group2)
    print(f"Levene's Test p-value: {levene_test.pvalue:.8f}")
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
    test_type = "Student's t-test"

    print(f"{test_type} Results:")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.8f}")
    
    
def PearsonCC_stat(feature_A, feature_B):

    # Create DataFrame
    df = pd.DataFrame({"Feature_A": feature_A, "Feature_B": feature_B})

    # Calculate Pearson correlation coefficient and p-value
    corr_coeff, p_value = pearsonr(df["Feature_A"], df["Feature_B"])

    print(f"Pearson Correlation Coefficient: {corr_coeff:.4f}")
    print(f"P-value: {p_value:.4f}")
    return corr_coeff, p_value



# Example data - replace these with your actual data
group1 = np.array([12, 15, 18, 22, 24, 16, 19, 20, 17, 21])
group2 = np.array([23, 25, 28, 32, 30, 27, 26, 29, 31, 24])


# Normality test (Shapiro-Wilk)
def check_normality(data, name):
    stat, p = stats.shapiro(data)
    print(f"\n{name} Normality:")
    print(f"Shapiro-Wilk p-value: {p:.8f}")
    return p > 0.05  # Return True if normal

def two_group_stat2(group1,name1, group2, name2):
    normal1 = check_normality(group1, name1)
    normal2 = check_normality(group2, name2)

    # Variance equality test (Levene's test)
    levene_stat, levene_p = stats.levene(group1, group2)
    print(f"\nLevene's Test for Equal Variances:")
    print(f"p-value: {levene_p:.8f}")
    equal_var = levene_p > 0.05

    # Hypothesis testing
    if normal1 and normal2:
        print("\nBoth groups are normally distributed")
        # Independent t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        print("\nIndependent t-test results:")
        print(f"t-statistic: {t_stat:.8f}")
        print(f"p-value: {p_value:.8f}")

        # Calculate Cohen's d effect size
        pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2)/2)
        cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        print(f"Cohen's d: {abs(cohen_d):.8f}")
    else:
        print("\nNon-normal distribution detected")
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(group1, group2)
        print("\nMann-Whitney U test results:")
        print(f"U-statistic: {u_stat}")
        print(f"p-value: {p_value:.8f}")

        # Calculate Cliff's delta effect size
        def cliffs_delta(a, b):
            dominance_matrix = np.sign(np.array(a)[:, None] - np.array(b))
            return np.mean(dominance_matrix)

        cd = cliffs_delta(group1, group2)
        print(f"Cliff's Delta: {abs(cd):.8f}")

        
def three_group_stat3(group1,name1, group2, name2, group3, name3):
    normal1 = check_normality(group1, name1)
    normal2 = check_normality(group2, name2)
    normal3 = check_normality(group3, name3)
    is_normal = all([normal1, normal2, normal3])
    stat, p = levene(group1, group2, group3)
    print(f"Levene's test: Statistic = {stat:.3f}, p-value = {p:.8f}")
    equal_var = p > 0.05
    if is_normal and equal_var:
        stat, p = f_oneway(group1, group2, group3)
        print(f"One-way ANOVA: F-statistic = {stat:.3f}, p-value = {p:.8f}")
        if p < 0.05:
            print("Significant difference between groups (p < 0.05). Performing post-hoc Tukey HSD...")
            # Combine data for Tukey HSD
            data = np.concatenate([group1, group2, group3])
            labels = ['Group1'] * len(group1) + ['Group2'] * len(group2) + ['Group3'] * len(group3)
            tukey = mc.MultiComparison(data, labels)
            tukey_result = tukey.tukeyhsd(alpha=0.05)
            print(tukey_result)
        else:
            print("No significant difference between groups.")
    else:
        # Use non-parametric Kruskal-Wallis test
        stat, p = kruskal(group1, group2, group3)
        print(f"Kruskal-Wallis test: Statistic = {stat:.3f}, p-value = {p:.8f}")
        if p < 0.05:
            print("Significant difference between groups (p < 0.05). Perform post-hoc Dunn's test.")
            data = [group1, group2, group3]
            dunn_result = sp.posthoc_dunn(data, p_adjust='bonferroni')
            print("\nDunn's test (Bonferroni-adjusted p-values):")
            print(dunn_result)
        else:
            print("No significant difference between groups.")