import cv2
import numpy as np
import numphly as nly
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
font = cv2.FONT_HERSHEY_SIMPLEX
rgb = (255, 255, 255)
EXP_LENGTH = 126000

def check_characters(input_string, target_characters):
    return set(target_characters).issubset(set(input_string))


def floor_tracker_xor_plus(idx_frames_f_bp_xor, loc_temp_floor, loc_temp_wall, loc_temp, conf_temp_wall, i_bp, thr_cf, thr_prox):
    
    frame_n = loc_temp_floor.shape[-1]
    l_bp_e = len(idx_frames_f_bp_xor)

    d_1_fw = -np.ones(l_bp_e)
    d_2_fw = -np.ones(l_bp_e)

    d_1_tc = -np.ones(l_bp_e)
    d_2_tc = -np.ones(l_bp_e)
    
    fly_tracker_f_bp_w  = -np.ones((2,frame_n))
    
    i_bp_e = 0

    for idx_i in idx_frames_f_bp_xor:
        

        
        if np.isnan(loc_temp_floor[0,0,0,idx_i]):
            loc_i = loc_temp_floor[1,0,:,idx_i]
            fly_id_floor = 1
        else:
            loc_i = loc_temp_floor[0,0,:,idx_i]
            fly_id_floor = 0
        ############# decide fly_id_floor value (could go wrong)############
#         if i_bp > 0:
#             d_1_tc[i_bp_e] = np.sqrt(np.sum(np.square(loc_f1-loc_i)))
#             d_2_tc[i_bp_e] = np.sqrt(np.sum(np.square(loc_f2-loc_i)))    
#             if np.isnan(d_1_tc[i_bp_e]) or np.isnan(d_2_tc[i_bp_e]):
#                 if ~np.isnan(d_1_tc[i_bp_e]):
#                     if d_1_tc[i_bp_e] > thr_prox:
#                         fly_id_floor = 1
#                     else:
#                         fly_id_floor = 0

#                 elif ~np.isnan(d_2_tc[i_bp_e]):
#                     if d_2_tc[i_bp_e] > thr_prox:
#                         fly_id_floor = 0
#                     else:
#                         fly_id_floor = 1
#             else:
#                 if d_1_tc[i_bp_e] >= d_2_tc[i_bp_e]:
#                     fly_id_floor = 1
#                 elif d_1_tc[i_bp_e] < d_2_tc[i_bp_e]:
#                     fly_id_floor = 0
         #####################################  
            
            
        loc_temp[fly_id_floor,i_bp,:,idx_i] = loc_i
        fly_tracker_f_bp_w[fly_id_floor,idx_i] = 0


        cf_w_bp_1   = conf_temp_wall[0,idx_i]
        cf_w_bp_2   = conf_temp_wall[1,idx_i]
        cf_w_bp_1_y = (cf_w_bp_1 >= thr_cf or np.isnan(cf_w_bp_1))
        cf_w_bp_2_y = (cf_w_bp_2 >= thr_cf or np.isnan(cf_w_bp_2))
        
 
        d_1_fw[i_bp_e] = np.sqrt(np.sum(np.square(loc_temp_wall [0,0,:,idx_i]-loc_i)))
        d_2_fw[i_bp_e] = np.sqrt(np.sum(np.square(loc_temp_wall [1,0,:,idx_i]-loc_i)))
        

        d_1_fw_yNan = np.isnan(d_1_fw[i_bp_e])
        d_2_fw_yNan = np.isnan(d_2_fw[i_bp_e])
        
        d_1_fw_y = d_1_fw[i_bp_e]>=thr_prox # distance to the other fly
        d_2_fw_y = d_2_fw[i_bp_e]>=thr_prox
        
        loc_fi    = np.nanmean(loc_temp[fly_id_floor,:,:,idx_i],axis=0)
        loc_fii    = np.nanmean(loc_temp[1-fly_id_floor,:,:,idx_i],axis=0)
        d_1_wtemp1 = np.sqrt(np.sum(np.square(loc_temp_wall [0,0,:,idx_i]-loc_fi)))
        d_1_wtemp2 = np.sqrt(np.sum(np.square(loc_temp_wall [0,0,:,idx_i]-loc_fii)))
        d_2_wtemp1 = np.sqrt(np.sum(np.square(loc_temp_wall [1,0,:,idx_i]-loc_fi)))
        d_2_wtemp2 = np.sqrt(np.sum(np.square(loc_temp_wall [1,0,:,idx_i]-loc_fii)))


         

        # the other fly is in side view
        if ~d_1_fw_yNan and ~d_2_fw_yNan:
            if d_1_fw_y and d_2_fw_y and cf_w_bp_1_y and cf_w_bp_2_y:
                if d_1_fw[i_bp_e] > d_2_fw[i_bp_e]:
                    loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]
                    fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 1
                else:
                    loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]

                    fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 2
            elif d_1_fw_y and cf_w_bp_1_y :
                
                d_1_wtemp1_yNan = np.isnan(d_1_wtemp1)
                d_1_wtemp2_yNan = np.isnan(d_1_wtemp2)

                if ~ d_1_wtemp1_yNan and ~d_1_wtemp2_yNan:
                    if d_1_wtemp1 > d_1_wtemp2:# good
                        loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]
                        fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 1

                else:
                    loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]
                    fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 1
                    
            elif d_2_fw_y and cf_w_bp_2_y:

                d_2_wtemp1_yNan = np.isnan(d_2_wtemp1)
                d_2_wtemp2_yNan = np.isnan(d_2_wtemp2)
                if ~ d_2_wtemp1_yNan and ~d_2_wtemp2_yNan:
                    if d_2_wtemp1 > d_2_wtemp2:# good
                        loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]
                        fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 2
                else:
                    loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]
                    fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 2

        elif ~d_1_fw_yNan:
            if d_1_fw_y and cf_w_bp_1_y:
                d_1_wtemp1_yNan = np.isnan(d_1_wtemp1)
                d_1_wtemp2_yNan = np.isnan(d_1_wtemp2)
                
                if ~ d_1_wtemp1_yNan and ~d_1_wtemp2_yNan:
                    if d_1_wtemp1 > d_1_wtemp2:# good
                        loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]
                        fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 1
                else:
                    loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]
                    fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 1
                
        elif ~d_2_fw_yNan:
  
            if d_2_fw_y and cf_w_bp_2_y:
                d_2_wtemp1_yNan = np.isnan(d_2_wtemp1)
                d_2_wtemp2_yNan = np.isnan(d_2_wtemp2)
                
                if ~ d_2_wtemp1_yNan and ~d_2_wtemp2_yNan:
                    if d_2_wtemp1 > d_2_wtemp2:# good
                        loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]
                        fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 2
                else:
                    loc_temp[1-fly_id_floor,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]
                    fly_tracker_f_bp_w[1-fly_id_floor,idx_i] = 2


        i_bp_e += 1
        
#         if idx_i == 59808:
            
#             print('loc_fi:', loc_fi)
#             print('loc_fii:', loc_fii)
#             print('loc_temp [0,i_bp,:,idx_i]:', loc_temp [0,0,:,idx_i])
#             print('loc_temp [1,i_bp,:,idx_i]:', loc_temp [1,0,:,idx_i])
#             print('loc_temp_floor [0,0,:,idx_i]:', loc_temp_floor [0,0,:,idx_i])
#             print('loc_temp_floor [1,0,:,idx_i]:', loc_temp_floor [1,0,:,idx_i])
#             print('loc_temp_wall [0,0,:,idx_i]:', loc_temp_wall [0,0,:,idx_i])
#             print('loc_temp_wall [1,0,:,idx_i]:', loc_temp_wall [1,0,:,idx_i])
#             print(d_1_wtemp1,  d_1_wtemp2, d_2_wtemp1,  d_2_wtemp2)
#             print(d_1_fw_yNan, d_1_fw_y,   d_2_fw_yNan, d_2_fw_y)
        
    return loc_temp, fly_tracker_f_bp_w


def wall_tracker_and(idx_frames_w_bp_and, loc_temp_wall, loc_temp, i_bp):
    
    frame_n = loc_temp_wall.shape[-1]
    
    
    l_bp_e = len(idx_frames_w_bp_and)
    
    i_bp_e = 0
    for idx_i in idx_frames_w_bp_and:
        
        loc_f1    = np.nanmean(loc_temp[0,:,:,idx_i],axis=0)
        loc_f2    = np.nanmean(loc_temp[1,:,:,idx_i],axis=0)
        
        loc_w1_bp = loc_temp_wall[0,0,:,idx_i]
        loc_w2_bp = loc_temp_wall[1,0,:,idx_i]
        d_f1_w1 = np.sqrt(np.sum(np.square(loc_w1_bp-loc_f1)))
        d_f1_w2 = np.sqrt(np.sum(np.square(loc_w2_bp-loc_f1)))
        d_f2_w1 = np.sqrt(np.sum(np.square(loc_w1_bp-loc_f2)))
        d_f2_w2 = np.sqrt(np.sum(np.square(loc_w2_bp-loc_f2)))
        
        d_i_f = np.ones(4)
        
        d_i_f[0] = d_f1_w1
        d_i_f[1] = d_f1_w2 #
        d_i_f[2] = d_f2_w1
        d_i_f[3] = d_f2_w2
        
        d_i_f[np.isnan(d_i_f)] = 1000
        idx_min = np.argmin(d_i_f)
        idx_max = np.argmax(d_i_f)

#         if idx_i == 96449:
#             print(d_i_f)
        if d_i_f[idx_min] == -1000:
            loc_temp[:,i_bp,:,idx_i] = loc_temp_wall[:,0,:,idx_i]

        elif idx_min==0 or idx_min==3:
            
            if idx_min==0 and idx_max == 3:
                loc_temp[0,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]
                loc_temp[1,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]     
            elif idx_max==0 and idx_min == 3:
                loc_temp[0,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]
                loc_temp[1,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]
            else:
                loc_temp[:,i_bp,:,idx_i] = loc_temp_wall[:,0,:,idx_i]
            
            
        elif idx_min==1 or idx_min==2:
            if idx_min==1 and idx_max == 2:
                loc_temp[:,i_bp,:,idx_i] = loc_temp_wall[:,0,:,idx_i]
                
            elif idx_max==1 and idx_min == 2:
                loc_temp[:,i_bp,:,idx_i] = loc_temp_wall[:,0,:,idx_i]
            else:
                loc_temp[0,i_bp,:,idx_i] = loc_temp_wall[1,0,:,idx_i]
                loc_temp[1,i_bp,:,idx_i] = loc_temp_wall[0,0,:,idx_i]

                
            
 
        
        i_bp_e += 1
    return loc_temp
def cross_tracker_and(idx_frames_c_bp_and, loc_temp_cross, loc_temp):
    
    frame_n = loc_temp_cross.shape[-1]
    l_bp_e = len(idx_frames_c_bp_and)

    i_bp_e = 0
    for idx_i in idx_frames_c_bp_and:
        
        loc_fw1    = loc_temp[0,0,:,idx_i]
        loc_fw2    = loc_temp[1,0,:,idx_i]
        
        loc_c1 = loc_temp_cross[0,0,:,idx_i]
        loc_c2 = loc_temp_cross[1,0,:,idx_i]
        
        d_fw1_c1 = np.sqrt(np.sum(np.square(loc_fw1 - loc_c1)))
        d_fw2_c1 = np.sqrt(np.sum(np.square(loc_fw2 - loc_c1)))
        d_fw1_c2 = np.sqrt(np.sum(np.square(loc_fw1 - loc_c2)))
        d_fw2_c2 = np.sqrt(np.sum(np.square(loc_fw2 - loc_c2)))
        
        d_i_f = np.ones(4)
        
        d_i_f[0] = d_fw1_c1
        d_i_f[1] = d_fw2_c1
        d_i_f[2] = d_fw1_c2
        d_i_f[3] = d_fw2_c2
        
        d_i_f[np.isnan(d_i_f)] = 1000
        idx_min = np.argmin(d_i_f)
        idx_max = np.argmax(d_i_f)
            
        if d_i_f[idx_min] == 1000:
            
            loc_temp[:,3,:,idx_i] = loc_temp_cross[:,0,:,idx_i]
            
        elif idx_min==0 or idx_min==3:
            
            loc_temp[:,3,:,idx_i] = loc_temp_cross[:,0,:,idx_i]
            
        elif idx_min==1 or idx_min==2:
            if idx_min==1 and idx_max == 2:
                loc_temp[:,3,:,idx_i] = loc_temp_cross[:,0,:,idx_i]
                
            elif idx_max==1 and idx_min == 2:
                loc_temp[:,3,:,idx_i] = loc_temp_cross[:,0,:,idx_i]
            else:
                loc_temp[0,3,:,idx_i] = loc_temp_cross[1,0,:,idx_i]
                loc_temp[1,3,:,idx_i] = loc_temp_cross[0,0,:,idx_i]

        i_bp_e += 1
    return loc_temp

def wall_tracker_xor_plus(idx_frames_w_bp_xor, loc_temp_wall, loc_temp, i_bp, thr_prox):
    frame_n = loc_temp_wall.shape[-1]
    l_bp_e = len(idx_frames_w_bp_xor)
    
    d_1_tc = -np.ones(l_bp_e)
    d_2_tc = -np.ones(l_bp_e)
    
    fly_tracker_w_bp_c = -np.ones((2,frame_n))
    i_bp_e = 0

    for idx_i in idx_frames_w_bp_xor:

        if np.isnan(loc_temp_wall[0,0,0,idx_i]):
            loc_i = loc_temp_wall[1,0,:,idx_i]
            fly_id_wall = 1
        else:
            loc_i = loc_temp_wall[0,0,:,idx_i]
            fly_id_wall = 0

        loc_temp_i = np.nanmean(loc_temp[:,0:3,:,idx_i],axis=1)
        
        

        
        d_1_tc[i_bp_e] = np.sqrt(np.sum(np.square(loc_temp_i[0,:]-loc_i)))
        d_2_tc[i_bp_e] = np.sqrt(np.sum(np.square(loc_temp_i[1,:]-loc_i)))

        if np.isnan(d_1_tc[i_bp_e]) or np.isnan(d_2_tc[i_bp_e]):
            if ~np.isnan(d_1_tc[i_bp_e]):
                if d_1_tc[i_bp_e] > thr_prox:
                    loc_temp[1, i_bp, :, idx_i] = loc_i
                    fly_tracker_w_bp_c[1,idx_i] = 0
                else:
                    loc_temp[0, i_bp, :, idx_i] = loc_i
                    fly_tracker_w_bp_c[0,idx_i] = 0
            elif ~np.isnan(d_2_tc[i_bp_e]):
                if d_2_tc[i_bp_e] > thr_prox:
                    loc_temp[0, i_bp, :, idx_i] = loc_i
                    fly_tracker_w_bp_c[0,idx_i] = 0
                else:
                    loc_temp[1, i_bp, :, idx_i] = loc_i
                    fly_tracker_w_bp_c[1,idx_i] = 0
            else:
                loc_temp[fly_id_wall, i_bp, :, idx_i] = loc_i
                fly_tracker_w_bp_c[fly_id_wall,idx_i] = 0
        else:
            if d_1_tc[i_bp_e] >= d_2_tc[i_bp_e]:

                loc_temp[1, i_bp, :, idx_i] = loc_i
                fly_tracker_w_bp_c[1,idx_i] = 0
            elif d_1_tc[i_bp_e] < d_2_tc[i_bp_e]:
                loc_temp[0, i_bp, :, idx_i] = loc_i
                fly_tracker_w_bp_c[0,idx_i] = 0
        i_bp_e += 1
    return loc_temp, fly_tracker_w_bp_c


def cross_tracker_xor_plus(idx_frames_w_bp_e, loc_temp, loc_temp_cross, conf_temp_cross, thr_prox):
    
    frame_n = loc_temp_cross.shape[-1]
    l_bp_e = len(idx_frames_w_bp_e)
    well_tracker_f_miss = np.zeros(frame_n) <0
    
    d_1_tc = -np.ones(l_bp_e)
    d_2_tc = -np.ones(l_bp_e)

    fly_tracker_fw_bp_c  = -np.ones((2,frame_n))

    i_bp_e = 0
    i_idx_miss = 0
    i_idx_miss_why = 0

    for idx_i in idx_frames_w_bp_e:

        if np.isnan(loc_temp_cross[0,0,0,idx_i]):
            loc_i = loc_temp_cross[1,0,:,idx_i]
            fly_id_cross = 1
        else:
            loc_i = loc_temp_cross[0,0,:,idx_i]
            fly_id_cross = 0


            
        loc_temp_i = np.nanmean(loc_temp[:,0:3,:,idx_i],axis=1)
        
        d_1_tc[i_bp_e] = np.sqrt(np.sum(np.square(loc_temp_i[0,:]-loc_i)))
        d_2_tc[i_bp_e] = np.sqrt(np.sum(np.square(loc_temp_i[1,:]-loc_i)))
#         if idx_i == 55006:
#             print(d_1_tc[i_bp_e],d_2_tc[i_bp_e])
#             print(loc_temp_i, loc_i)
        if np.isnan(d_1_tc[i_bp_e]) or np.isnan(d_2_tc[i_bp_e]):
            if np.isnan(d_2_tc[i_bp_e]):
                if d_1_tc[i_bp_e] > thr_prox:
                    loc_temp[1, 3, :, idx_i] = loc_i
                    fly_tracker_fw_bp_c[1,idx_i] = fly_id_cross+1
                else:
                    loc_temp[0, 3, :, idx_i] = loc_i
                    fly_tracker_fw_bp_c[0,idx_i] = fly_id_cross+1
            else:
                if d_2_tc[i_bp_e] > thr_prox:
                    loc_temp[0, 3, :, idx_i] = loc_i
                    fly_tracker_fw_bp_c[0,idx_i] = fly_id_cross+1
                else:
                    loc_temp[1, 3, :, idx_i] = loc_i
                    fly_tracker_fw_bp_c[1,idx_i] = fly_id_cross+1
        else:
            if d_1_tc[i_bp_e] < d_2_tc[i_bp_e]:

                loc_temp[0, 3, :, idx_i] = loc_i
                fly_tracker_fw_bp_c[0,idx_i] = fly_id_cross+1
            else:
                loc_temp[1, 3, :, idx_i] = loc_i
                fly_tracker_fw_bp_c[1,idx_i] = fly_id_cross+1


        i_bp_e += 1
    return loc_temp, fly_tracker_fw_bp_c


def bodyparts_fandw(loc_floor, loc_wall, conf_floor, conf_wall):
    frame_n = 126000
    loc_temp = np.zeros((2,3,2,frame_n))
    loc_temp[:,:,:,:]      = np.NaN
    conf_temp    = np.zeros((2,3,frame_n))
    tracker_temp = np.zeros((2,3,frame_n))
    for i_bp_f in range(3):
        loc_floor_temp = np.zeros((2,1,2,frame_n))
        loc_floor_temp[:,:,:,:]      = np.NaN
        loc_floor_temp[:,0,:,:] = loc_floor[:,i_bp_f,:,:]
        loc_wall_temp = np.zeros((2,1,2,frame_n))
        loc_wall_temp[:,:,:,:]       = np.NaN
        loc_wall_temp[:,0,:,:]  = loc_wall[:,i_bp_f,:,:]
        fly_tracker_floor   = nly.tracking_analysis_or(loc_floor_temp)
        fly_tracker_wall    = nly.tracking_analysis_or(loc_wall_temp)

        fly_tracker = np.sum(fly_tracker_floor, axis=0)+np.sum(fly_tracker_wall, axis=0)


        idx_frames = np.arange(frame_n)
        tracker_12 = -np.ones(frame_n)
        for idx_i_fw in idx_frames:

            if fly_tracker[idx_i_fw]>=2:
                loc_1_m_i = loc_floor[:,i_bp_f,:,idx_i_fw]
                loc_2_m_i = loc_wall [:,i_bp_f,:,idx_i_fw]
                conf_f_i  = conf_floor[:,i_bp_f,idx_i_fw]
                conf_w_i  = conf_wall[:,i_bp_f,idx_i_fw]
                
                loc_temp[:,i_bp_f,:,idx_i_fw], conf_temp[:,i_bp_f,idx_i_fw], tracker_temp[:,i_bp_f,idx_i_fw], tracker_12[idx_i_fw] = tracker_selection(loc_1_m_i, loc_2_m_i, conf_f_i, conf_w_i)
    return loc_temp




def bodyparts_1and2(loc_1, loc_2, conf_1, conf_2):
    frame_n = 126000
    
    i_bp_f = 0
    loc_temp  = np.zeros((2,1,2,frame_n))
    loc_temp[:,:,:,:] = np.NaN    
    
    conf_temp    = np.zeros((2,frame_n))
    tracker_temp = np.zeros((2,frame_n))
    
    loc_1_m = np.mean(loc_1, axis = 1)
    loc_2_m = np.mean(loc_2, axis = 1)
    
    conf_1_bp = np.mean(conf_1, axis = 1)
    conf_2_bp = np.mean(conf_2, axis = 1)
    
    loc_1_temp = np.zeros((2,1,2,frame_n))
    loc_1_temp[:,:,:,:] = np.NaN
    loc_1_temp[:,0,:,:] = loc_1_m
    loc_2_temp = np.zeros((2,1,2,frame_n))
    loc_2_temp[:,:,:,:] = np.NaN
    loc_2_temp[:,0,:,:] = loc_2_m

    fly_tracker_1   = nly.tracking_analysis_or(loc_1_temp)
    fly_tracker_2   = nly.tracking_analysis_or(loc_2_temp)

    fly_tracker = np.sum(fly_tracker_1, axis=0)+np.sum(fly_tracker_2, axis=0)

    
    idx_frames = np.arange(frame_n)
    tracker_12 = -np.ones((frame_n))
    for idx_i_fw in idx_frames:
        if fly_tracker[idx_i_fw]>=2:
            loc_1_m_i = loc_1_temp[:,0,:,idx_i_fw]
            loc_2_m_i = loc_2_temp[:,0,:,idx_i_fw]
            conf_1_i  = conf_1_bp[:,idx_i_fw]
            conf_2_i  = conf_2_bp[:,idx_i_fw]
            loc_temp[:,i_bp_f,:,idx_i_fw], conf_temp[:,idx_i_fw], tracker_temp[:,idx_i_fw], tracker_12[idx_i_fw] = tracker_selection(loc_1_m_i, loc_2_m_i, conf_1_i, conf_2_i)
                

    return loc_temp



def save_image_fwc(cap, loc_floor_i, loc_wall_i, loc_cross_i, 
                   conf_f_i, conf_w_i, conf_c_i, video, idx_image, image, d_xy, bodyparts_floor):
    color_floor = [(0,0,255),(0,255,0),]
    color_wall  = [(0,255,255),(255,255,0),]
    color_cross = [(255,0,0),(255,0,255)]
    n_ID = loc_wall_i.shape[0]
    f = 0;w = 0;c = 0;
    yyyy = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_image)
    res, frame = cap.read()
    for i_bp in range(3):
        for i_fly in range(n_ID):
            dot_xy = (loc_floor_i[i_fly,i_bp,:]-d_xy).astype('int')
            try:
                cv2.circle(frame, dot_xy, 2, color_floor[i_fly], i_bp+1)
                cv2.putText(frame, 'f'+ str(i_fly) + ' ' + "%.2f" %conf_f_i[i_fly,i_bp]+' '+bodyparts_floor[i_bp], 
                        (500,480+i_bp*10+i_fly*30), font, .35, color_floor[i_fly])
                f = 1
            except:
                xxxx = 1


    for i_bp in range(3):
        for i_fly in range(n_ID):
            dot_xy = (loc_wall_i[i_fly,i_bp,:]-d_xy).astype('int')
            try:
                cv2.circle(frame, dot_xy, i_bp+1, color_wall[i_fly], 1)
                cv2.putText(frame, 'w'+ str(i_fly) + ' ' + "%.2f" %conf_w_i[i_fly,i_bp]+' '+bodyparts_floor[i_bp], 
                        (500,10+i_bp*10+i_fly*30), font, .35, color_wall[i_fly])
                w = 1
            except:
                xxxx = 1


    for i_fly in range(n_ID):
        dot_xy = (loc_cross_i[i_fly,0,:]-d_xy).astype('int')
        try:
            cv2.circle(frame, dot_xy, 2, color_cross[i_fly], i_bp+1)
            cv2.putText(frame, 'c'+ str(i_fly) + ' ' +"%.2f" %conf_c_i[i_fly], 
                        (520,200+i_fly*10), font, .35, color_cross[i_fly])
            c = 1
        except:
            xxxx = 1

    cv2.imwrite(image + '_' +str(f)+str(w)+str(c)+'_rec.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
def save_image_temp(cap, loc_i, fly_tracker_temp_i, video, idx_image, image, d_xy, bodyparts_floor):
    #             red       green     blue      white
    color_temp = ((0,0,255),(0,255,0),(255,0,0),(255,255,255))
    tracker    = ['n','f','w','cr']
    bodyparts  = ['h','c','t','cr']
    yyyy = 0
    n_ID = fly_tracker_temp_i.shape[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_image)
    res, frame = cap.read()
    for i_bp in range(4):
        for i_fly in range(n_ID):
            tracker_id = (fly_tracker_temp_i[i_fly,i_bp]).astype(int)
            dot_xy = (loc_i[i_fly,i_bp,:]-d_xy).astype('int')
            try:
                cv2.circle(frame, dot_xy, (i_bp+1)+(i_fly*2), color_temp[tracker_id], 1)
                cv2.circle(frame, (120,5+i_bp*15+i_fly*70), (i_bp+1)+(i_fly*2), color_temp[tracker_id], 1)
                cv2.putText(frame, 
                            tracker[tracker_id]+' '+str(tracker_id)+' '+str(i_fly)+' '+bodyparts[i_bp]+' '+str(dot_xy[0])+' '+str(dot_xy[1]),
                            (5, 10+i_bp*15+i_fly*70), font, .35, color_temp[tracker_id])
                yyyy = 1
            except:
                xxxx = 1

    if yyyy == 1:
        cv2.imwrite(image + '_temp_rec.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#     else:
#         cv2.imwrite(image + '_temp_rec_No_tracking.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])






def convert_to_new_coords(p1, p2, point):
    x_translated = point[0,:] - p1[0,:]
    y_translated = point[1,:] - p1[1,:]

    # Calculate the direction vector from p1 to p2
    dx = p2[0,:] - p1[0,:]
    dy = p2[1,:] - p1[1,:]

    # Compute the rotation angle to align the vector with the y-axis
    theta = np.arctan2(dx, dy)

    # Calculate rotation matrix components
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Apply the rotation
    new_x = x_translated * cos_theta - y_translated * sin_theta
    new_y = x_translated * sin_theta + y_translated * cos_theta
    point_new = np.vstack((new_x,new_y))
    return point_new



def reverse_conversion(p1, p2, new_point):
    # Calculate the direction vector from p1 to p2
    dx = p2[0,:] - p1[0,:]
    dy = p2[1,:] - p1[1,:]
    
    # Compute the original rotation angle
    theta = np.arctan2(dx, dy)
    
    # Calculate inverse rotation matrix components
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Apply inverse rotation
    original_x_translated = new_point[0,:] * cos_theta + new_point[1,:] * sin_theta
    original_y_translated = -new_point[0,:] * sin_theta + new_point[1,:] * cos_theta
    
    # Translate back to original coordinate system
    original_x = original_x_translated + p1[0,:]
    original_y = original_y_translated + p1[1,:]
    point_new = np.vstack((original_x,original_y))
    return point_new


def bodylength(loc, thr_center, thr_space, mode, parameter_mm2p, thr_l_size):
    
    l1 = np.NaN 
    l2 = np.NaN
    l1_hc = np.NaN 
    l2_hc = np.NaN
    l1_ct = np.NaN 
    l2_ct = np.NaN
    
    well_tracker_eql = nly.tracking_analysis_eql(loc, 1, 1)
    
    Head1   = np.squeeze(loc[0,0,:,:])
    center1 = np.squeeze(loc[0,1,:,:])
    Tail1   = np.squeeze(loc[0,2,:,:])
    
    Head2   = np.squeeze(loc[1,0,:,:])
    center2 = np.squeeze(loc[1,1,:,:])
    Tail2   = np.squeeze(loc[1,2,:,:])
    

    loc_center = np.ones((2,loc.shape[-1]))*240
    d_f1_c = np.sqrt(np.sum(np.square(center1-loc_center),axis=0))
    d_f2_c = np.sqrt(np.sum(np.square(center2-loc_center),axis=0))

    d_H1T1 = np.sqrt(np.sum(np.square(Head1-Tail1),axis=0))
    d_H2T2 = np.sqrt(np.sum(np.square(Head2-Tail2),axis=0))
    d_H1C1 = np.sqrt(np.sum(np.square(Head1-center1),axis=0))
    d_H2C2 = np.sqrt(np.sum(np.square(Head2-center2),axis=0))
    d_C1T1 = np.sqrt(np.sum(np.square(Tail1-center1),axis=0))
    d_C2T2 = np.sqrt(np.sum(np.square(Tail2-center2),axis=0))
    
    d_H1H2 = np.sqrt(np.sum(np.square(Head1-Head2),axis=0))
    d_T1T2 = np.sqrt(np.sum(np.square(Tail1-Tail2),axis=0))

    d_H1T2 = np.sqrt(np.sum(np.square(Head1-Tail2),axis=0))
    d_H2T1 = np.sqrt(np.sum(np.square(Head2-Tail1),axis=0))
    well_tracker_HH_TT = np.logical_and(d_H1H2 > thr_space, d_T1T2 > thr_space)
    well_tracker_HT_HT = np.logical_and(d_H1T2 > thr_space, d_H2T1 > thr_space)
    if mode == 'top':
        well_tracker_C_C   = np.logical_and(d_f1_c <= thr_center, d_f2_c <=thr_center)
    elif mode == 'side':
        well_tracker_C_C   = np.logical_and(d_f1_c >= thr_center, d_f2_c >=thr_center)
        
    well_tracker_size = np.logical_and(well_tracker_HH_TT, well_tracker_HT_HT)
    well_tracker_size = np.logical_and(well_tracker_C_C,   well_tracker_size)
    well_tracker_size = np.logical_and(well_tracker_eql,   well_tracker_size)

    gaps_l, gaps = nly.tracked_segments(well_tracker_size)
    n_gaps = len(gaps)
    if n_gaps > 0:
        start_size = gaps[np.argmax(gaps_l)]
        l_size = np.max(gaps_l)
        if l_size >= thr_l_size:
#             midpoint = int(l_size/2+start_size)
#             idx_gap_max = np.arange(midpoint-30,midpoint+30)
            idx_gap_max = np.arange(start_size+2,start_size+l_size-2)
            l1    = np.mean(d_H1T1[idx_gap_max])*parameter_mm2p  
            l2    = np.mean(d_H2T2[idx_gap_max])*parameter_mm2p
            l1_hc = np.mean(d_H1C1[idx_gap_max])*parameter_mm2p
            l2_hc = np.mean(d_H2C2[idx_gap_max])*parameter_mm2p
            l1_ct = np.mean(d_C1T1[idx_gap_max])*parameter_mm2p  
            l2_ct = np.mean(d_C2T2[idx_gap_max])*parameter_mm2p
        else:
            start_size = 0
            l_size = 0
#             idx_gap_max = np.arange(start_size+1,start_size+l_size-2)
#             l1 = np.mean(d_H1T1[idx_gap_max])*parameter_mm2p  
#             l2 = np.mean(d_H2T2[idx_gap_max])*parameter_mm2p
    else:
        start_size = 0
        l_size = 0
    return l1, l2, l1_hc, l2_hc, l1_ct, l2_ct, start_size, l_size


def tracker_selection(loc_1_m_i, loc_2_m_i, conf_1_i, conf_2_i):
    tracker_12 = -1
    loc_temp_i  = np.zeros((2,2))
    loc_temp_i[:,:] = np.NaN    
    conf_temp_i    = np.zeros(2)
    tracker_temp_i = np.zeros(2)
    
    conf_12_i_11 = conf_1_i[0] >= conf_2_i[0]
    conf_12_i_22 = conf_1_i[1] >= conf_2_i[1]
    conf_12_i_12 = conf_1_i[0] >= conf_2_i[1]
    conf_12_i_21 = conf_1_i[1] >= conf_2_i[0]

    conf_1_i_1 = conf_1_i[0]==1
    conf_1_i_2 = conf_1_i[1]==1
    conf_2_i_1 = conf_2_i[0]==1
    conf_2_i_2 = conf_2_i[1]==1
    conf_11_i_12 = conf_1_i[0] >= conf_1_i[1]
    conf_22_i_12 = conf_2_i[0] >= conf_2_i[1]

    d_loc = np.zeros(6)
    c_loc = np.zeros((6,2))
    loc_i = np.zeros((2,2,6))

    d_loc[0] = np.sqrt(np.sum(np.square(loc_1_m_i[0,:]-loc_1_m_i[1,:])))
    d_loc[1] = np.sqrt(np.sum(np.square(loc_2_m_i[0,:]-loc_2_m_i[1,:])))
    d_loc[2] = np.sqrt(np.sum(np.square(loc_1_m_i[0,:]-loc_2_m_i[0,:])))
    d_loc[3] = np.sqrt(np.sum(np.square(loc_1_m_i[0,:]-loc_2_m_i[1,:])))
    d_loc[4] = np.sqrt(np.sum(np.square(loc_1_m_i[1,:]-loc_2_m_i[0,:])))
    d_loc[5] = np.sqrt(np.sum(np.square(loc_1_m_i[1,:]-loc_2_m_i[1,:])))

    idx_sort_d_loc = np.argsort(d_loc)

    d_loc_sorted = d_loc[idx_sort_d_loc]
    idx_sort_d_loc_nan = np.isnan(d_loc_sorted)
    folder_name = ''

    for idx_d in range(6):
        if idx_sort_d_loc_nan[idx_d]:
            folder_name += 'X'
        else:
            folder_name += str(idx_sort_d_loc[idx_d])

    first2 = folder_name[0:2]
    first3 = folder_name[0:3]
    last3  = folder_name[3:6]


    if first2 == "25" or first2 == "52":
        tracker_12 = 1
        if conf_12_i_11:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0

        else:
            loc_temp_i[0,:] = loc_2_m_i[0,:];
            conf_temp_i[0] = conf_2_i[0]
            tracker_temp_i[0] = 1

        if conf_12_i_22:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[1,:];
            conf_temp_i[1] = conf_2_i[1]
            tracker_temp_i[1] = 1

    elif first2 == "34" or first2 == "43":
        tracker_12 = 2
        if conf_12_i_12:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[1,:];
            conf_temp_i[0] = conf_2_i[1]
            tracker_temp_i[0] = 1

        if conf_12_i_21:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[0,:];
            conf_temp_i[1] = conf_2_i[0]
            tracker_temp_i[1] = 1

    elif folder_name == "0XXXXX":
        tracker_12 = 3
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0

        loc_temp_i[1,:] = loc_1_m_i[1,:];
        conf_temp_i[1] = conf_1_i[1]
        tracker_temp_i[1] = 0

    elif folder_name == "1XXXXX":
        tracker_12 = 4
        loc_temp_i[0,:] = loc_2_m_i[0,:];
        conf_temp_i[0] = conf_2_i[0]
        tracker_temp_i[0] = 1

        loc_temp_i[1,:] = loc_2_m_i[1,:];
        conf_temp_i[1] = conf_2_i[0]
        tracker_temp_i[1] = 1

    elif folder_name == "2XXXXX":
        tracker_12 = 5
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0

        loc_temp_i[1,:] = loc_2_m_i[0,:];
        conf_temp_i[1] = conf_2_i[0]
        tracker_temp_i[1] = 1

    elif folder_name == "3XXXXX":
        tracker_12 = 6
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0

        loc_temp_i[1,:] = loc_2_m_i[1,:];
        conf_temp_i[1] = conf_2_i[1]
        tracker_temp_i[1] = 1
    elif folder_name == "4XXXXX":
        tracker_12 = 7
        loc_temp_i[0,:] = loc_1_m_i[1,:];
        conf_temp_i[0] = conf_1_i[1]
        tracker_temp_i[0] = 0

        loc_temp_i[1,:] = loc_2_m_i[0,:];
        conf_temp_i[1] = conf_2_i[0]
        tracker_temp_i[1] = 1

    elif folder_name == "5XXXXX":
        tracker_12 = 8
        loc_temp_i[0,:] = loc_1_m_i[1,:];
        conf_temp_i[0] = conf_1_i[1]
        tracker_temp_i[0] = 0

        loc_temp_i[1,:] = loc_2_m_i[1,:];
        conf_temp_i[1] = conf_2_i[1]
        tracker_temp_i[1] = 1
    elif first3 == "024" or first3 == "042": # distance filter

        if last3 == "XXX":
            if conf_1_i[0] >= 0.9 and conf_2_i[0] >= 0.9 and d_loc[2]>=20 and d_loc[4]>=20:
                tracker_12 = 9

                loc_temp_i[0,:] = loc_1_m_i[0,:];
                conf_temp_i[0] = conf_1_i[0]
                tracker_temp_i[0] = 0

                loc_temp_i[1,:] = loc_2_m_i[0,:];
                conf_temp_i[1] = conf_2_i[0]
                tracker_temp_i[1] = 1
        else:
            tracker_12 = 9
            loc_temp_i[1,:] = loc_2_m_i[1,:];
            conf_temp_i[1] = conf_2_i[0]
            tracker_temp_i[1] = 1
            if conf_12_i_11:
                loc_temp_i[0,:] = loc_1_m_i[0,:];
                conf_temp_i[0] = conf_1_i[0]
                tracker_temp_i[0] = 0
            else:
                loc_temp_i[0,:] = loc_2_m_i[0,:];
                conf_temp_i[0] = conf_2_i[0]
                tracker_temp_i[0] = 1

    elif first3 == "035" or first3 == "053": # distance filter

        if last3 == "XXX":
            if conf_1_i[0] >= 0.9 and conf_2_i[0] >= 0.9 and d_loc[3]>=20 and d_loc[5]>=20:
                tracker_12 = 10
                loc_temp_i[0,:] = loc_1_m_i[0,:];
                conf_temp_i[0] = conf_1_i[0]
                tracker_temp_i[0] = 0
                loc_temp_i[1,:] = loc_2_m_i[1,:];
                conf_temp_i[1] = conf_2_i[0]
                tracker_temp_i[1] = 1
        else:
            tracker_12 = 10
            loc_temp_i[1,:] = loc_2_m_i[0,:];
            conf_temp_i[1] = conf_2_i[0]
            tracker_temp_i[1] = 1
            if conf_12_i_12:
                loc_temp_i[0,:] = loc_1_m_i[0,:];
                conf_temp_i[0] = conf_1_i[0]
                tracker_temp_i[0] = 0
            else:
                loc_temp_i[0,:] = loc_2_m_i[1,:];
                conf_temp_i[0] = conf_2_i[0]
                tracker_temp_i[0] = 1

    elif first3 == "123" or first3 == "132":
        if last3 == "XXX":
            if conf_1_i[0] >= 0.9 and conf_2_i[0] >= 0.9 and d_loc[3]>=20 and d_loc[2]>=20:
                tracker_12 = 11
                loc_temp_i[0,:] = loc_1_m_i[0,:];
                conf_temp_i[0] = conf_1_i[0]
                tracker_temp_i[0] = 0
                loc_temp_i[1,:] = loc_2_m_i[0,:];
                conf_temp_i[1] = conf_2_i[0]
                tracker_temp_i[1] = 1
        else:
            tracker_12 = 11

            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[0]
            tracker_temp_i[1] = 0

            if conf_12_i_11:
                loc_temp_i[0,:] = loc_1_m_i[0,:];
                conf_temp_i[0] = conf_1_i[0]
                tracker_temp_i[0] = 0
            else:
                loc_temp_i[0,:] = loc_2_m_i[0,:];
                conf_temp_i[0] = conf_2_i[0]
                tracker_temp_i[0] = 1

    elif first3 == "145" or first3 == "154":
        if last3 == "XXX":
            if conf_1_i[1] >= 0.9 and conf_2_i[1] >= 0.9 and d_loc[4]>=20 and d_loc[5]>=20:
                tracker_12 = 12
                loc_temp_i[0,:] = loc_2_m_i[0,:];
                conf_temp_i[0] = conf_2_i[0]
                tracker_temp_i[0] = 1
                loc_temp_i[1,:] = loc_1_m_i[1,:];
                conf_temp_i[1] = conf_1_i[1]
                tracker_temp_i[1] = 0
        else:
            tracker_12 = 12
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
            if conf_12_i_21:
                loc_temp_i[1,:] = loc_1_m_i[1,:];
                conf_temp_i[1] = conf_1_i[1]
                tracker_temp_i[1] = 0
            else:
                loc_temp_i[1,:] = loc_2_m_i[0,:];
                conf_temp_i[1] = conf_2_i[0]
                tracker_temp_i[1] = 1

    elif folder_name == "240XXX" or folder_name == "204XXX":
        tracker_12 = 13
        loc_temp_i[1,:] = loc_1_m_i[1,:];
        conf_temp_i[1] = conf_1_i[1]
        tracker_temp_i[1] = 0
        if conf_12_i_11:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0

        else:
            loc_temp_i[0,:] = loc_2_m_i[0,:];
            conf_temp_i[0] = conf_2_i[0]
            tracker_temp_i[1] = 1

    elif folder_name == "213XXX" or folder_name == "231XXX":
        tracker_12 = 14
        loc_temp_i[1,:] = loc_2_m_i[1,:];
        conf_temp_i[1] = conf_2_i[1]
        tracker_temp_i[1] = 1
        if conf_12_i_11:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[0,:];
            conf_temp_i[0] = conf_2_i[1]
            tracker_temp_i[0] = 1

    elif folder_name == "305XXX" or folder_name == "350XXX":
        tracker_12 = 15
        loc_temp_i[1,:] = loc_1_m_i[1,:];
        conf_temp_i[1] = conf_1_i[1]
        tracker_temp_i[1] = 0
        if conf_12_i_12:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[1,:];
            conf_temp_i[0] = conf_2_i[1]
            tracker_temp_i[0] = 1
    elif folder_name == "312XXX" or folder_name == "321XXX":
        tracker_12 = 16
        loc_temp_i[1,:] = loc_2_m_i[0,:];
        conf_temp_i[1] = conf_2_i[1]
        tracker_temp_i[1] = 1
        if conf_12_i_12:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[1,:];
            conf_temp_i[0] = conf_2_i[1]
            tracker_temp_i[0] = 1
    elif folder_name == "402XXX" or folder_name == "420XXX":
        tracker_12 = 17
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0
        if conf_12_i_21:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[0,:];
            conf_temp_i[1] = conf_2_i[0]
            tracker_temp_i[1] = 1
    elif folder_name == "415XXX" or folder_name == "451XXX":
        tracker_12 = 18
        loc_temp_i[0,:] = loc_2_m_i[1,:];
        conf_temp_i[0] = conf_2_i[1]
        tracker_temp_i[0] = 1
        if conf_12_i_21:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[0,:];
            conf_temp_i[1] = conf_2_i[0]
            tracker_temp_i[1] = 1

    elif folder_name == "503XXX" or folder_name == "530XXX":
        tracker_12 = 19
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0
        if conf_12_i_22:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[1,:];
            conf_temp_i[1] = conf_2_i[1]
            tracker_temp_i[1] = 1
    elif folder_name == "514XXX" or folder_name == "541XXX":
        tracker_12 = 20
        loc_temp_i[0,:] = loc_2_m_i[0,:];
        conf_temp_i[0] = conf_2_i[0]
        tracker_temp_i[0] = 1
        if conf_12_i_22:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[1,:];
            conf_temp_i[1] = conf_2_i[1]
            tracker_temp_i[1] = 1
    elif check_characters(last3, '045'):
        tracker_12 = 21
        loc_temp_i[1,:] = loc_1_m_i[1,:];
        conf_temp_i[1] = conf_1_i[1]
        tracker_temp_i[1] = 0
        if conf_12_i_11:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[0,:];
            conf_temp_i[0] = conf_2_i[0]
            tracker_temp_i[0] = 1

    elif check_characters(last3, '023'):
        tracker_12 = 22
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0

        if conf_12_i_22:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[1,:];
            conf_temp_i[1] = conf_2_i[1]
            tracker_temp_i[1] = 1
    elif check_characters(last3, '124'):
        tracker_12 = 23
        loc_temp_i[1,:] = loc_2_m_i[0,:];
        conf_temp_i[1] = conf_2_i[0]
        tracker_temp_i[1] = 1
        if conf_12_i_12:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[1,:];
            conf_temp_i[0] = conf_2_i[1]
            tracker_temp_i[0] = 1
    elif check_characters(last3, '135'):
        tracker_12 = 24
        loc_temp_i[1,:] = loc_2_m_i[1,:];
        conf_temp_i[1] = conf_2_i[1]
        tracker_temp_i[1] = 1
        if conf_12_i_11:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[0,:];
            conf_temp_i[0] = conf_2_i[0]
            tracker_temp_i[0] = 1

    elif folder_name[0] == '2':
        tracker_12 = 25
        loc_temp_i[1,:] = loc_1_m_i[1,:];
        conf_temp_i[1] = conf_1_i[1]
        tracker_temp_i[1] = 0
        if conf_12_i_11:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[0,:];
            conf_temp_i[0] = conf_2_i[0]
            tracker_temp_i[0] = 1

    elif folder_name[0] == '3':
        tracker_12 = 26
        loc_temp_i[1,:] = loc_2_m_i[0,:];
        conf_temp_i[1] = conf_2_i[0]
        tracker_temp_i[1] = 1
        if conf_12_i_12:
            loc_temp_i[0,:] = loc_1_m_i[0,:];
            conf_temp_i[0] = conf_1_i[0]
            tracker_temp_i[0] = 0
        else:
            loc_temp_i[0,:] = loc_2_m_i[1,:];
            conf_temp_i[0] = conf_2_i[1]
            tracker_temp_i[0] = 1
    elif folder_name[0] == '4':
        tracker_12 = 27
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0
        if conf_12_i_21:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[0,:];
            conf_temp_i[1] = conf_2_i[0]
            tracker_temp_i[1] = 1
    elif folder_name[0] == '5':
        tracker_12 = 28
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0
        if conf_12_i_22:
            loc_temp_i[1,:] = loc_1_m_i[1,:];
            conf_temp_i[1] = conf_1_i[1]
            tracker_temp_i[1] = 0
        else:
            loc_temp_i[1,:] = loc_2_m_i[1,:];
            conf_temp_i[1] = conf_2_i[1]
            tracker_temp_i[1] = 1
    elif folder_name[0] == '1' or folder_name[5] == '0':
        tracker_12 = 30
        loc_temp_i[0,:] = loc_1_m_i[0,:];
        conf_temp_i[0] = conf_1_i[0]
        tracker_temp_i[0] = 0
        loc_temp_i[1,:] = loc_1_m_i[1,:];
        conf_temp_i[1] = conf_1_i[1]
        tracker_temp_i[1] = 0
    return loc_temp_i, conf_temp_i, tracker_temp_i, tracker_12