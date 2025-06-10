import cv2
import itertools
import numpy as np
import numphly as nly
from pathlib import Path
import os
# import scipy.stats as stats
import pandas as pd
# from statsmodels.stats.multicomp import pairwise_tukeyhsd


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
font = cv2.FONT_HERSHEY_SIMPLEX
rgb = (255, 255, 255)

arena_parameters = {
    'well_radius': 8.4,  # mm
    'food_cup_radius': 3.5,  # mm
}


def all_tracking_data(h5_dir, loaded_size, filter_status, scorer_floor, scorer_wall, scorer_cross, experiments_mscore, experiments_index, fileloc_set, videofile_set, animalID, bodyparts_floor, num_file, num_well, num_exp, window_size, crop_h, frame_n, frame_n_e, thr_presence):

    l_ws = len(window_size)
    
    d_ht = loaded_size['d_ht_tracked']
    d_hc = loaded_size['d_hc_tracked']
    d_ct = loaded_size['d_ct_tracked']
    
    d_ht_total = np.sum(d_ht,axis=1)
    d_hc_total = np.sum(d_hc,axis=1)
    d_ct_total = np.sum(d_ct,axis=1)
    
    d_f1f2_m_tracked = np.zeros((l_ws,num_exp))
    t_f1f2_m_tracked = np.zeros((l_ws,num_exp))
    
    ad_d_f1f2_m_tracked = np.zeros((l_ws,num_exp))
    t_ad_d_f1f2_m_tracked = np.zeros((l_ws,num_exp))
    
    t_f1f2_c_tracked = np.zeros((l_ws,num_exp,2))
    
    d_f1f2_tracked = np.zeros((l_ws,num_exp,3))
    t_f1f2_tracked = np.zeros((l_ws,num_exp,3))

    
    d_tracked  = np.zeros((l_ws,num_exp))
    t_speed_tracked = np.zeros((l_ws,num_exp))
    
    norm_f1_f2_chase_m = np.zeros((l_ws,num_exp))
    norm_f1_f2_face_m  = np.zeros((l_ws,num_exp))
    t_chase_tracked  = np.zeros((l_ws,num_exp))
    t_face_tracked  = np.zeros((l_ws,num_exp))

    norm_f1_f2_chase_m_wocr = np.zeros((l_ws,num_exp))
    norm_f1_f2_face_m_wocr  = np.zeros((l_ws,num_exp))
    t_chase_wocr_tracked  = np.zeros((l_ws,num_exp))
    t_face_wocr_tracked  = np.zeros((l_ws,num_exp))
    

    
    
    
    

    T_exp = frame_n_e/60
    ii = 0
    for i in range(num_file):
        experiment = experiments_mscore[fileloc_set[i]]
        sexes        = experiment['sex']
        genotypes   = experiment['genotype']
        experiment_index = experiments_index[i]
        start_t = np.max(experiment['starting_time'])
    
        
        for j in range(num_well):
            sex = sexes[j]
            genotype = genotypes[j]

            if sex != -1:
                circle = crop_h[ii,:]
                parameter_mm2p = (arena_parameters['well_radius']/circle[2])
                well = 'v_'+experiment_index+'_'+videofile_set[j]
                
                hfile_wall = h5_dir+'/'+well+scorer_wall+'_el'+filter_status+'.h5'
                my_file_wall = Path(hfile_wall)
                hfile_floor = h5_dir+'/'+well+scorer_floor+'_el'+filter_status+'.h5'
                my_file_floor = Path(hfile_floor)
                hfile_cross = h5_dir+'/'+well+scorer_cross+'_el'+filter_status+'.h5'
                my_file_cross = Path(hfile_cross)

                wall_floor = my_file_wall.is_file() and my_file_floor.is_file()
                wall_floor_cross = wall_floor and my_file_cross.is_file()
                if wall_floor_cross:

                    npz_file_name = h5_dir+'/'+well+'_el'+filter_status+'.npz'
                    loaded_data = np.load(npz_file_name)
                    loc_temp          = loaded_data['loc_temp']

                    i_l = 0 
                    for win_size in window_size:
                        frame_n_ds = (frame_n/win_size)
                        start_t_ds = np.ceil(start_t/win_size)
                        frame_n_e_ds = np.ceil(frame_n_e/win_size)
                        well_tracker_select_ds = np.logical_and(np.arange(frame_n_ds)>=start_t_ds,np.arange(frame_n_ds)<start_t_ds+frame_n_e_ds)
                        well_tracker_select_ds_NaN = ~well_tracker_select_ds
                        
                        loc_temp_ds = nly.downsample(loc_temp,win_size,thr_presence)
                        
                        loc_temp_ds[:,:,:,well_tracker_select_ds_NaN] = np.NaN
                        
                        loc_m_ht = np.mean(loc_temp_ds[:,[0,2],:,:], axis=1)
                        loc_m_c  = loc_temp_ds[:,1,:,:]
                        loc_m_cr = loc_temp_ds[:,3,:,:]

                        loc_m_parts = np.stack([loc_m_ht, loc_m_c, loc_m_cr], axis=1)
                    
#                         tracker = nly.tracking_analysis_and(loc_m_parts)

                        #fly pair distance
                        loc_temp_ds_m = np.nanmean(loc_m_parts, axis=1)
                        fly_tracker_temp_f1 = loc_temp_ds[0,:,0,:]>=0
                        fly_tracker_temp_f2 = loc_temp_ds[1,:,0,:]>=0
                        well_tracker_f1f2_temp_bp = np.logical_and(fly_tracker_temp_f1, fly_tracker_temp_f2)
                        well_tracker_f1f2_temp_2bp = np.sum(well_tracker_f1f2_temp_bp[0:3,:],axis=0)>=2
                        fly_tracker_f1_temp_2bp = np.sum(fly_tracker_temp_f1[0:3,:], axis=0)>=2
                        fly_tracker_f2_temp_2bp = np.sum(fly_tracker_temp_f2[0:3,:], axis=0)>=2
                        well_tracker_f1f2_temp_2bpcr = np.logical_and(fly_tracker_f1_temp_2bp,
                                                                      fly_tracker_temp_f2[3,:])
                        well_tracker_f1f2_temp_cr2bp = np.logical_and(fly_tracker_f2_temp_2bp,
                                                                      fly_tracker_temp_f1[3,:])
                        
                        well_tracker_f1f2_temp = np.logical_or(well_tracker_f1f2_temp_cr2bp,
                                                               well_tracker_f1f2_temp_2bpcr)
                        well_tracker_f1f2_temp = np.logical_or(well_tracker_f1f2_temp,
                                                               well_tracker_f1f2_temp_2bp)

                        d_f1f2_m   = np.squeeze(np.sqrt(np.sum(np.square(np.diff(
                            loc_temp_ds_m,axis=0)),axis=1)))
                        
                        

                        t_f1f2_m_tracked[i_l,ii] = (np.sum(well_tracker_f1f2_temp)/frame_n_e_ds)
                        d_f1f2_m_tracked[i_l,ii] = np.mean(d_f1f2_m[well_tracker_f1f2_temp])*parameter_mm2p
                        
                        
                        #fly pair total distance change
                        ad_d_f1f2_m = np.abs(np.diff(d_f1f2_m))
                        well_tacker_ad_d_f1f2_m = ad_d_f1f2_m>=0
                        
                        t_ad_d_f1f2_m_tracked[i_l,ii] = (np.sum(well_tacker_ad_d_f1f2_m)/frame_n_e_ds)*T_exp
                        ad_d_f1f2_m_tracked[i_l,ii]   = np.mean(ad_d_f1f2_m[well_tacker_ad_d_f1f2_m])*parameter_mm2p
                        
                        loc_center = np.ones((2,loc_temp_ds_m.shape[-1]))*240
                        #fly to center distance
                        d_f1_c   = np.sqrt(np.sum(np.square(loc_temp_ds_m[0,:,:]-loc_center),axis=0))
                        d_f2_c   = np.sqrt(np.sum(np.square(loc_temp_ds_m[1,:,:]-loc_center),axis=0))

                        thr_center = 5/parameter_mm2p

                        well_tracker_wall   = np.logical_or (d_f1_c>=thr_center, d_f2_c>=thr_center)
                        well_tracker_center = np.logical_and(d_f1_c <thr_center,d_f2_c <thr_center)

                        t_f1f2_c_tracked[i_l,ii,0] = np.sum(well_tracker_wall)
                        t_f1f2_c_tracked[i_l,ii,1] = np.sum(well_tracker_center)

                        #fly pair body parts distance
                        d_f1f2    = np.sqrt(np.sum(np.square(loc_temp_ds[0,:,:,:]-loc_temp_ds[1,:,:,:]),axis=1))
                        
                        well_tracker_f1f2 = (d_f1f2>=0)[0:3]
                        
                        d_f1f2_tracked[i_l,ii,0] = np.mean(d_f1f2[0,well_tracker_f1f2[0,:]])*parameter_mm2p
                        d_f1f2_tracked[i_l,ii,1] = np.mean(d_f1f2[1,well_tracker_f1f2[1,:]])*parameter_mm2p
                        d_f1f2_tracked[i_l,ii,2] = np.mean(d_f1f2[2,well_tracker_f1f2[2,:]])*parameter_mm2p
                        
                        t_f1f2_tracked[i_l,ii,:] = (np.sum(well_tracker_f1f2, axis=1)/frame_n_e_ds)*T_exp

                        #speed
                        d_loc_df = np.sqrt(np.sum(np.square(np.diff(loc_temp_ds_m,axis=2)),axis=1)) 
                        well_tracker = np.logical_and(d_loc_df[0,:]>=0, d_loc_df[1,:]>=0)
                        time = np.sum(well_tracker)
                        d_tracked[i_l,ii]        = np.mean(np.sum(d_loc_df[:,well_tracker],axis=1))*parameter_mm2p
                        t_speed_tracked[i_l,ii]  = (time/frame_n_e_ds)*T_exp
                        


                        d_ht_norm = d_ht_total[ii]/parameter_mm2p
                        d_hc_norm = d_hc_total[ii]/parameter_mm2p
                        d_ct_norm = d_ct_total[ii]/parameter_mm2p
                        
                        # face
                        norm_f1_f2_face, norm_f1_f2_face_wocr = calculate_norm_f1_f2_aggression(
                            loc_temp_ds, d_f1f2, d_ht_norm, d_hc_norm, d_ct_norm)

                        norm_f1_f2_face_m_wocr[i_l,ii] = np.nanmean(norm_f1_f2_face_wocr)
                        t_face_wocr_tracked[i_l,ii]    = np.sum(
                            ~np.isnan(norm_f1_f2_face_wocr))/frame_n_e_ds*T_exp
                        
                        norm_f1_f2_face_m[i_l,ii]      = np.nanmean(norm_f1_f2_face)
                        t_face_tracked[i_l,ii]         = np.sum(
                            ~np.isnan(norm_f1_f2_face))/frame_n_e_ds*T_exp 


                        # chase
                        norm_f1_f2_chase, norm_f1_f2_chase_wocr = calculate_norm_f1_f2_pursuit(
                            loc_temp_ds, d_ht_norm, d_hc_norm, d_ct_norm)
                        
                        norm_f1_f2_chase_m_wocr[i_l,ii] = np.nanmean(norm_f1_f2_chase_wocr)
                        t_face_wocr_tracked[i_l,ii]    = np.sum(
                            ~np.isnan(norm_f1_f2_chase_wocr))/frame_n_e_ds*T_exp
                        
                        norm_f1_f2_chase_m[i_l,ii] = np.nanmean(norm_f1_f2_chase)
                        t_chase_tracked[i_l,ii] = np.sum(
                            ~np.isnan(norm_f1_f2_chase))/frame_n_e_ds*T_exp
                        
                        print(well, win_size,
                              "%.3f" %(np.sum(well_tracker_f1f2_temp)/frame_n_e_ds),
                              "%.3f" %(np.sum(~np.isnan(norm_f1_f2_face))/frame_n_e_ds),
                              "%.3f" %(np.sum(~np.isnan(norm_f1_f2_chase))/frame_n_e_ds),
                             )
                        i_l += 1
                else:
                    print('no h files')
                ii = ii+1
    return d_f1f2_m_tracked, t_f1f2_m_tracked, ad_d_f1f2_m_tracked, t_ad_d_f1f2_m_tracked, t_f1f2_c_tracked, d_f1f2_tracked, t_f1f2_tracked, d_tracked, t_speed_tracked,norm_f1_f2_face_m, norm_f1_f2_chase_m, t_face_tracked, t_chase_tracked,norm_f1_f2_face_m_wocr,norm_f1_f2_chase_m_wocr, t_face_wocr_tracked, t_chase_wocr_tracked









def all_tracking_data_rate(h5_dir, loaded_size, filter_status, scorer_floor, scorer_wall, scorer_cross, experiments_mscore, experiments_index, fileloc_set, videofile_set, animalID, bodyparts_floor, bodyparts_wall, bodyparts_cross, num_file, num_well, num_exp, window_size, crop_h, frame_n, frame_n_e, thr_presence):

    l_ws = len(window_size)
    
    d_ht = loaded_size['d_ht_tracked']
    d_hc = loaded_size['d_hc_tracked']
    d_ct = loaded_size['d_ct_tracked']
    
    numBP_floor = len(bodyparts_floor)
    numBP_wall = len(bodyparts_wall)
    numBP_cross = len(bodyparts_cross)
    well_done_rate_temp_2bp_selected = np.zeros((l_ws,num_exp))
    well_done_rate_temp_selected = np.zeros((l_ws,num_exp))
    well_done_rate_f_2bp_selected = np.zeros((l_ws,num_exp))
    well_done_rate_w_2bp_selected = np.zeros((l_ws,num_exp))
    well_done_rate_c_selected = np.zeros((l_ws,num_exp))
    T_exp = frame_n_e/60
    ii = 0
    for i in range(num_file):
        experiment = experiments_mscore[fileloc_set[i]]
        sexes        = experiment['sex']
        genotypes   = experiment['genotype']
        experiment_index = experiments_index[i]
        start_t = np.max(experiment['starting_time'])
    
        
        for j in range(num_well):
            sex = sexes[j]
            genotype = genotypes[j]

            if sex != -1:
                circle = crop_h[ii,:]
                parameter_mm2p = (arena_parameters['well_radius']/circle[2])
                well = 'v_'+experiment_index+'_'+videofile_set[j]
                
                hfile_wall = h5_dir+'/'+well+scorer_wall+'_el'+filter_status+'.h5'
                my_file_wall = Path(hfile_wall)
                hfile_floor = h5_dir+'/'+well+scorer_floor+'_el'+filter_status+'.h5'
                my_file_floor = Path(hfile_floor)
                hfile_cross = h5_dir+'/'+well+scorer_cross+'_el'+filter_status+'.h5'
                my_file_cross = Path(hfile_cross)

                wall_floor = my_file_wall.is_file() and my_file_floor.is_file()
                wall_floor_cross = wall_floor and my_file_cross.is_file()
                if wall_floor_cross:
                    Dataframe_floor = pd.read_hdf(os.path.join(hfile_floor))
                    Dataframe_wall  = pd.read_hdf(os.path.join(hfile_wall))
                    Dataframe_cross = pd.read_hdf(os.path.join(hfile_cross))

                    loc_floor, conf_floor = nly.read_loc(Dataframe_floor, scorer_floor, animalID, bodyparts_floor)
                    loc_wall,  conf_wall  = nly.read_loc(Dataframe_wall,  scorer_wall,  animalID, bodyparts_wall)
                    loc_cross, conf_cross = nly.read_loc(Dataframe_cross, scorer_cross, animalID, bodyparts_cross)
                    
                    thr_cf = 0.95
                    
                    fly_tracker_f = nly.tracking_analysis_or_cf(loc_floor,conf_floor,thr_cf)
                    fly_tracker_f_n = np.zeros((2,numBP_floor,2,frame_n))==1
                    fly_tracker_f_n[:,:,0,:] = ~fly_tracker_f
                    fly_tracker_f_n[:,:,1,:] = ~fly_tracker_f
                    loc_floor[fly_tracker_f_n] = np.NaN

                    
                    fly_tracker_w = nly.tracking_analysis_or_cf(loc_wall, conf_wall,thr_cf)
                    fly_tracker_w_n = np.zeros((2,numBP_wall,2,frame_n))==1
                    fly_tracker_w_n[:,:,0,:] = ~fly_tracker_w
                    fly_tracker_w_n[:,:,1,:] = ~fly_tracker_w
                    loc_wall[fly_tracker_w_n]  = np.NaN
                    
                    fly_tracker_c = nly.tracking_analysis_or_cf(loc_cross,conf_cross,thr_cf)
                    fly_tracker_c_n = np.zeros((2,numBP_cross,2,frame_n))==1
                    fly_tracker_c_n[:,:,0,:] = ~fly_tracker_c
                    fly_tracker_c_n[:,:,1,:] = ~fly_tracker_c
                    loc_cross[fly_tracker_c_n]  = np.NaN
                    conf_temp_cross = np.zeros((2,1,frame_n))
                    conf_temp_cross[:,0,:] = np.mean(conf_cross, axis=1)
                    fly_tracker_c_bp       = conf_temp_cross[:,0,:]>=thr_cf
                    
                    
                    
                    npz_file_name = h5_dir+'/'+well+'_el'+filter_status+'.npz'
                    loaded_data = np.load(npz_file_name)
                    loc_temp          = loaded_data['loc_temp']
                    
                    
                    loc_temp_floor          = np.zeros((2,1,2,frame_n))
                    loc_temp_floor[:,:,:,:] = np.NaN

                    loc_temp_wall          = np.zeros((2,1,2,frame_n))
                    loc_temp_wall[:,:,:,:] = np.NaN

                    loc_temp_cross          = np.zeros((2,1,2,frame_n))
                    loc_temp_cross[:,:,:,:] = np.NaN
                    loc_temp_cross[:,0,:,:] = np.mean(loc_cross, axis=1)
                    
                    
                    
                    
                    well_tracke_f_eql = ~nly.tracking_analysis_eql(loc_floor, 1, 1)
                    well_tracke_w_eql = ~nly.tracking_analysis_eql(loc_wall, 1, 1)
                    well_tracke_c_eql = ~nly.tracking_analysis_eql(loc_temp_cross, 1, 1)
                    
                    loc_floor[:,:,:,well_tracke_f_eql] = np.NaN
                    loc_wall [:,:,:,well_tracke_w_eql] = np.NaN
                    loc_temp_cross [:,:,:,well_tracke_c_eql] = np.NaN
                    print(well, np.sum(well_tracke_f_eql),np.sum(well_tracke_w_eql))
                    
                    i_l = 0 
                    for win_size in window_size:
                        frame_n_ds = (frame_n/win_size)
                        start_t_ds = np.ceil(start_t/win_size)
                        frame_n_e_ds = np.ceil(frame_n_e/win_size)
                        well_tracker_select_ds = np.logical_and(np.arange(frame_n_ds)>=start_t_ds,np.arange(frame_n_ds)<start_t_ds+frame_n_e_ds)
                        well_tracker_select_ds_NaN = ~well_tracker_select_ds
                        
                        
                        
                        loc_temp_ds       = nly.downsample(loc_temp,  win_size,thr_presence)
                        loc_f_ds          = nly.downsample(loc_floor, win_size,thr_presence)
                        loc_w_ds          = nly.downsample(loc_wall,  win_size,thr_presence)
                        loc_temp_cross_ds = nly.downsample(loc_temp_cross,  win_size,thr_presence)
                        
                        loc_temp_ds[:,:,:,well_tracker_select_ds_NaN] = np.NaN
                        loc_f_ds   [:,:,:,well_tracker_select_ds_NaN] = np.NaN
                        loc_w_ds   [:,:,:,well_tracker_select_ds_NaN] = np.NaN
                        loc_temp_cross_ds[:,:,:,well_tracker_select_ds_NaN] = np.NaN

                        fly_tracker_temp_f1 = loc_temp_ds[0,:,0,:]>=0
                        fly_tracker_temp_f2 = loc_temp_ds[1,:,0,:]>=0
                        fly_tracker_f_f1 = loc_f_ds[0,:,0,:]>=0
                        fly_tracker_f_f2 = loc_f_ds[1,:,0,:]>=0
                        fly_tracker_w_f1 = loc_w_ds[0,:,0,:]>=0
                        fly_tracker_w_f2 = loc_w_ds[1,:,0,:]>=0
                        fly_tracker_c_f1 = loc_temp_cross_ds[0,:,0,:]>=0
                        fly_tracker_c_f2 = loc_temp_cross_ds[1,:,0,:]>=0
                        
                        well_tracker_f1f2_temp_bp = np.logical_and(fly_tracker_temp_f1, fly_tracker_temp_f2)
                        well_tracker_f1f2_f_bp    = np.logical_and(fly_tracker_f_f1, fly_tracker_f_f2)
                        well_tracker_f1f2_w_bp    = np.logical_and(fly_tracker_w_f1, fly_tracker_w_f2)
                        well_tracker_f1f2_c_bp    = np.logical_and(fly_tracker_c_f1, fly_tracker_c_f2)
                        
                        well_tracker_f1f2_temp_2bp = np.sum(well_tracker_f1f2_temp_bp[0:3,:],axis=0)>=2
                        well_tracker_f1f2_f_2bp    = np.sum(well_tracker_f1f2_f_bp,   axis=0)>=2
                        well_tracker_f1f2_w_2bp    = np.sum(well_tracker_f1f2_w_bp,   axis=0)>=2
                        
                        fly_tracker_f1_temp_2bp = np.sum(fly_tracker_temp_f1[0:3,:], axis=0)>=2
                        fly_tracker_f2_temp_2bp = np.sum(fly_tracker_temp_f2[0:3,:], axis=0)>=2
                        well_tracker_f1f2_temp_2bpcr = np.logical_and(fly_tracker_f1_temp_2bp,
                                                                      fly_tracker_temp_f2[3,:])
                        well_tracker_f1f2_temp_cr2bp = np.logical_and(fly_tracker_f2_temp_2bp,
                                                                      fly_tracker_temp_f1[3,:])
                        
                        well_tracker_f1f2_temp = np.logical_or(well_tracker_f1f2_temp_cr2bp,
                                                               well_tracker_f1f2_temp_2bpcr)
                        well_tracker_f1f2_temp = np.logical_or(well_tracker_f1f2_temp,
                                                               well_tracker_f1f2_temp_2bp)
                        
                        well_done_rate_temp_selected[i_l,ii]    = np.sum(well_tracker_f1f2_temp)/frame_n_e_ds
                        well_done_rate_temp_2bp_selected[i_l,ii]= np.sum(well_tracker_f1f2_temp_2bp)/frame_n_e_ds
                        well_done_rate_f_2bp_selected[i_l,ii]   = np.sum(well_tracker_f1f2_f_2bp)/frame_n_e_ds
                        well_done_rate_w_2bp_selected[i_l,ii]   = np.sum(well_tracker_f1f2_w_2bp)/frame_n_e_ds
                        
                        well_done_rate_c_selected[i_l,ii]   = np.sum(well_tracker_f1f2_c_bp)/frame_n_e_ds
                        print(well, win_size, frame_n_e_ds,
#                               well_tracker_f1f2_temp_2bp.shape,
#                               well_tracker_f1f2_f_2bp.shape,
#                               well_tracker_f1f2_w_2bp.shape,
#                               well_tracker_f1f2_c.shape,
                              "%.3f" %well_done_rate_f_2bp_selected[i_l,ii],
                              "%.3f" %well_done_rate_w_2bp_selected[i_l,ii],
                              "%.3f" %well_done_rate_c_selected[i_l,ii],
                              "%.3f" %well_done_rate_temp_2bp_selected[i_l,ii],
                              "%.3f" %well_done_rate_temp_selected[i_l,ii],

                             )
                        i_l += 1
                else:
                    print('no h files')
                ii = ii+1
    return well_done_rate_f_2bp_selected, well_done_rate_w_2bp_selected, well_done_rate_c_selected, well_done_rate_temp_2bp_selected, well_done_rate_temp_selected



def events_tracking_rate(event_ts, event_type, sex, buffer, thr, well_tracker, well_tracker_select):
    
    n_event_sel_trk_f     = np.zeros(4)
    n_event_sel_trk_m_L   = np.zeros(4)
    n_event_sel_trk_m_UWE = np.zeros(4)
    n_event_sel_trk_f[:]     = np.NaN
    n_event_sel_trk_m_L[:]   = np.NaN
    n_event_sel_trk_m_UWE[:] = np.NaN
    well_tracker_selected = np.logical_and(well_tracker, well_tracker_select)
    if sex == 0:
        if sum(event_type == 1)>0:
            event_ts_HB      = event_ts[event_type==1]
            n_event_selected = sum(well_tracker_select[event_ts_HB])
            n_event          = len(event_ts_HB)
            if n_event > 0:
                count_event_tracker = np.zeros(n_event)
                for dt in np.arange(-buffer,buffer+1):
                    event_tracker = well_tracker[event_ts_HB+dt]
                    count_event_tracker += event_tracker
                event_tracker_dt = count_event_tracker >= thr
                event_ts_HB_tracked = event_ts_HB[event_tracker_dt]
                n_event_sel_trk_f[0] = sum(well_tracker_selected[event_ts_HB_tracked])
                n_event_sel_trk_f[1] = sum(well_tracker[event_ts_HB_tracked])
                n_event_sel_trk_f[2] = n_event_selected
                n_event_sel_trk_f[3] = n_event
#                 n_event_sel_trk_f[0] = n_event_sel_trk_f[0] + sum(well_tracker_selected[event_ts_HB_tracked])
#                 n_event_sel_trk_f[1] = n_event_sel_trk_f[1] + sum(well_tracker[event_ts_HB_tracked])
#                 n_event_sel_trk_f[2] = n_event_sel_trk_f[2] + n_event_selected
#                 n_event_sel_trk_f[3] = n_event_sel_trk_f[3] + n_event

    elif sex == 1:
        if sum(event_type == 3)>0:
            event_ts_L       = event_ts[event_type==3]
            n_event_selected = sum(well_tracker_select[event_ts_L])
            n_event          = len(event_ts_L)
            if n_event > 0:
                count_event_tracker = np.zeros(n_event)
                for dt in np.arange(-buffer,buffer+1):
                    event_tracker = well_tracker[event_ts_L+dt]
                    count_event_tracker += event_tracker
                event_tracker_dt = count_event_tracker >= thr
                event_ts_L_tracked = event_ts_L[event_tracker_dt]
#                 n_event_sel_trk_m_L[0] = n_event_sel_trk_m_L[0] + sum(well_tracker_selected[event_ts_L_tracked])
#                 n_event_sel_trk_m_L[1] = n_event_sel_trk_m_L[1] + sum(well_tracker[event_ts_L_tracked])
#                 n_event_sel_trk_m_L[2] = n_event_sel_trk_m_L[2] + n_event_selected
#                 n_event_sel_trk_m_L[3] = n_event_sel_trk_m_L[3] + n_event
                n_event_sel_trk_m_L[0] = sum(well_tracker_selected[event_ts_L_tracked])
                n_event_sel_trk_m_L[1] = sum(well_tracker[event_ts_L_tracked])
                n_event_sel_trk_m_L[2] = n_event_selected
                n_event_sel_trk_m_L[3] = n_event
        if sum(event_type == 2)>0:
            event_ts_UWE     = event_ts[event_type==2]
            n_event_selected = sum(well_tracker_select[event_ts_UWE])
            n_event          = len(event_ts_UWE)
            if n_event > 0:
                count_event_tracker = np.zeros(n_event)
                for dt in np.arange(-buffer,buffer+1):
                    event_tracker = well_tracker[event_ts_UWE+dt]
                    count_event_tracker += event_tracker
                event_tracker_dt = count_event_tracker >= thr
                event_ts_UWE_tracked = event_ts_UWE[event_tracker_dt]
#                 n_event_sel_trk_m_UWE[0] = n_event_sel_trk_m_UWE[0] + sum(well_tracker_selected[event_ts_UWE_tracked])
#                 n_event_sel_trk_m_UWE[1] = n_event_sel_trk_m_UWE[1] + sum(well_tracker[event_ts_UWE_tracked])
#                 n_event_sel_trk_m_UWE[2] = n_event_sel_trk_m_UWE[2] + n_event_selected
#                 n_event_sel_trk_m_UWE[3] = n_event_sel_trk_m_UWE[3] + n_event
                n_event_sel_trk_m_UWE[0] = sum(well_tracker_selected[event_ts_UWE_tracked])
                n_event_sel_trk_m_UWE[1] = sum(well_tracker[event_ts_UWE_tracked])
                n_event_sel_trk_m_UWE[2] = n_event_selected
                n_event_sel_trk_m_UWE[3] = n_event

    return n_event_sel_trk_f, n_event_sel_trk_m_L, n_event_sel_trk_m_UWE

def calculate_norm_f1_f2_pursuit(loc_temp_ds, d_ht_norm, d_hc_norm, d_ct_norm):
    norm_f1_f2_chase         = np.zeros(loc_temp_ds.shape[-1])
    norm_f1_f2_chase[:]      = np.NaN
    
    norm_f1_f2_chase_wocr    = np.zeros(loc_temp_ds.shape[-1])
    norm_f1_f2_chase_wocr[:] = np.NaN
    
    d_f1f2_H1C2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,0,:,:]-loc_temp_ds[1,1,:,:]),axis=0))
    d_f1f2_H2C1 = np.sqrt(np.sum(np.square(loc_temp_ds[0,1,:,:]-loc_temp_ds[1,0,:,:]),axis=0))
    d_f1f2_H1T2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,0,:,:]-loc_temp_ds[1,2,:,:]),axis=0))
    d_f1f2_H2T1 = np.sqrt(np.sum(np.square(loc_temp_ds[0,2,:,:]-loc_temp_ds[1,0,:,:]),axis=0))
    d_f1f2_C1T2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,1,:,:]-loc_temp_ds[1,2,:,:]),axis=0))
    d_f1f2_C2T1 = np.sqrt(np.sum(np.square(loc_temp_ds[0,2,:,:]-loc_temp_ds[1,1,:,:]),axis=0))

    d_f1f2_H1Cr2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,0,:,:]-loc_temp_ds[1,3,:,:]),axis=0))
    d_f1f2_C1Cr2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,1,:,:]-loc_temp_ds[1,3,:,:]),axis=0))
    d_f1f2_Cr2T1 = np.sqrt(np.sum(np.square(loc_temp_ds[0,2,:,:]-loc_temp_ds[1,3,:,:]),axis=0))
    d_f1f2_H2Cr1 = np.sqrt(np.sum(np.square(loc_temp_ds[1,0,:,:]-loc_temp_ds[0,3,:,:]),axis=0))
    d_f1f2_C2Cr1 = np.sqrt(np.sum(np.square(loc_temp_ds[1,1,:,:]-loc_temp_ds[0,3,:,:]),axis=0))
    d_f1f2_Cr1T2 = np.sqrt(np.sum(np.square(loc_temp_ds[1,2,:,:]-loc_temp_ds[0,3,:,:]),axis=0))

    d_f1f2_Cr1Cr2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,3,:,:]-loc_temp_ds[1,3,:,:]),axis=0))

    d_d_f1_f2_h1t2_h2t1 = d_f1f2_H1T2-d_f1f2_H2T1
    well_tracker_h_t = ~np.isnan(d_d_f1_f2_h1t2_h2t1)
    d_d_f1_f2_h1c2_h2c1 = d_f1f2_H1C2-d_f1f2_H2C1
    well_tracker_h_c = ~np.isnan(d_d_f1_f2_h1c2_h2c1)
    d_d_f1_f2_c1t2_c2t1 = d_f1f2_C1T2-d_f1f2_C2T1
    well_tracker_c_t = ~np.isnan(d_d_f1_f2_c1t2_c2t1)

    d_d_f1_f2_h1cr2_h2cr1 = d_f1f2_H1Cr2-d_f1f2_H2Cr1
    well_tracker_h_cr = ~np.isnan(d_d_f1_f2_h1cr2_h2cr1)
    d_d_f1_f2_cr1t2_cr2t1 = d_f1f2_Cr1T2-d_f1f2_Cr2T1
    well_tracker_cr_t = ~np.isnan(d_d_f1_f2_cr1t2_cr2t1)         


    d_d_f1_f2_h1cr2_cr2t1 = d_f1f2_H1Cr2-d_f1f2_Cr2T1
    well_tracker_ht1_cr2 = ~np.isnan(d_d_f1_f2_h1cr2_cr2t1)
    d_d_f1_f2_h1cr2_cr2c1 = d_f1f2_H1Cr2-d_f1f2_C1Cr2
    well_tracker_hc1_cr2 = ~np.isnan(d_d_f1_f2_h1cr2_cr2c1)
    d_d_f1_f2_c1cr2_cr2t1 = d_f1f2_C1Cr2-d_f1f2_Cr2T1
    well_tracker_ct1_cr2 = ~np.isnan(d_d_f1_f2_c1cr2_cr2t1)

    d_d_f1_f2_h2cr1_cr1t2 = d_f1f2_H2Cr1-d_f1f2_Cr1T2
    well_tracker_ht2_cr1 = ~np.isnan(d_d_f1_f2_h2cr1_cr1t2)
    d_d_f1_f2_h2cr1_cr1c2 = d_f1f2_H2Cr1-d_f1f2_C2Cr1
    well_tracker_hc2_cr1 = ~np.isnan(d_d_f1_f2_h2cr1_cr1c2)
    d_d_f1_f2_c2cr1_cr1t2 = d_f1f2_C2Cr1-d_f1f2_Cr1T2
    well_tracker_ct2_cr1 = ~np.isnan(d_d_f1_f2_c2cr1_cr1t2)
    
    well_tracker_cr1_cr2 = ~np.isnan(d_f1f2_Cr1Cr2)
    
    idx_norm_f1_f2_chase = np.isnan(norm_f1_f2_chase)
    norm_f1_f2_chase[well_tracker_h_t] = np.abs(d_d_f1_f2_h1t2_h2t1[well_tracker_h_t]/d_ht_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_h_t)

    well_tracker_h_c_chase = np.logical_and(idx_norm_f1_f2_chase,well_tracker_h_c)
    norm_f1_f2_chase[well_tracker_h_c_chase] = np.abs(d_d_f1_f2_h1c2_h2c1[well_tracker_h_c_chase]/d_hc_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_h_c_chase)

    well_tracker_c_t_chase = np.logical_and(idx_norm_f1_f2_chase,well_tracker_c_t)
    norm_f1_f2_chase[well_tracker_c_t_chase] = np.abs(d_d_f1_f2_c1t2_c2t1[well_tracker_c_t_chase]/d_ct_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_c_t_chase)

    well_tracker_h_cr_chase = np.logical_and(idx_norm_f1_f2_chase,well_tracker_h_cr)
    norm_f1_f2_chase[well_tracker_h_cr_chase] = np.abs(d_d_f1_f2_h1cr2_h2cr1[well_tracker_h_cr_chase]/d_hc_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_h_cr_chase)

    well_tracker_cr_t_chase = np.logical_and(idx_norm_f1_f2_chase,well_tracker_cr_t)
    norm_f1_f2_chase[well_tracker_cr_t_chase] = np.abs(d_d_f1_f2_cr1t2_cr2t1[well_tracker_cr_t_chase]/d_ct_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_cr_t_chase)


    norm_f1_f2_chase_wocr[:] = norm_f1_f2_chase

    well_tracker_ht1_cr2_chase = np.logical_and(idx_norm_f1_f2_chase, well_tracker_ht1_cr2)
    norm_f1_f2_chase[well_tracker_ht1_cr2_chase] = np.abs(d_d_f1_f2_h1cr2_cr2t1[well_tracker_ht1_cr2_chase]/d_ht_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_ht1_cr2_chase)

    well_tracker_hc1_cr2_chase = np.logical_and(idx_norm_f1_f2_chase, well_tracker_hc1_cr2)
    norm_f1_f2_chase[well_tracker_hc1_cr2_chase] = np.abs(d_d_f1_f2_h1cr2_cr2c1[well_tracker_hc1_cr2_chase]/d_hc_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_hc1_cr2_chase)

    well_tracker_ct1_cr2_chase = np.logical_and(idx_norm_f1_f2_chase, well_tracker_ct1_cr2)
    norm_f1_f2_chase[well_tracker_ct1_cr2_chase] = np.abs(d_d_f1_f2_c1cr2_cr2t1[well_tracker_ct1_cr2_chase]/d_ct_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_ct1_cr2_chase)

    well_tracker_ht2_cr1_chase = np.logical_and(idx_norm_f1_f2_chase, well_tracker_ht2_cr1)
    norm_f1_f2_chase[well_tracker_ht2_cr1_chase] = np.abs(d_d_f1_f2_h2cr1_cr1t2[well_tracker_ht2_cr1_chase]/d_ht_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_ht2_cr1_chase)

    well_tracker_hc2_cr1_chase = np.logical_and(idx_norm_f1_f2_chase, well_tracker_hc2_cr1)
    norm_f1_f2_chase[well_tracker_hc2_cr1_chase] = np.abs(d_d_f1_f2_h2cr1_cr1c2[well_tracker_hc2_cr1_chase]/d_hc_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_hc2_cr1_chase)

    well_tracker_ct2_cr1_chase = np.logical_and(idx_norm_f1_f2_chase, well_tracker_ct2_cr1)
    norm_f1_f2_chase[well_tracker_ct2_cr1_chase] = np.abs(d_d_f1_f2_c2cr1_cr1t2[well_tracker_ct2_cr1_chase]/d_ct_norm)
    idx_norm_f1_f2_chase = np.logical_and(idx_norm_f1_f2_chase, ~well_tracker_ct2_cr1_chase)
    
    well_tracker_cr1_cr2_chase = np.logical_and(idx_norm_f1_f2_chase, well_tracker_cr1_cr2)
    norm_f1_f2_chase[well_tracker_ct2_cr1_chase] = 0
    
    return norm_f1_f2_chase, norm_f1_f2_chase_wocr

def calculate_norm_f1_f2_aggression(loc_temp_ds, d_f1f2, d_ht_norm, d_hc_norm, d_ct_norm):
    
    norm_f1_f2_face         = np.zeros(d_f1f2.shape[-1])
    norm_f1_f2_face[:]      = np.NaN
    
    norm_f1_f2_face_wocr    = np.zeros(d_f1f2.shape[-1])
    norm_f1_f2_face_wocr[:] = np.NaN
    
    d_d_f1f2_hh_tt = d_f1f2[2,:]-d_f1f2[0,:]
    well_tracker_ht = ~np.isnan(d_d_f1f2_hh_tt)
    d_d_f1f2_hh_cc = d_f1f2[1,:]-d_f1f2[0,:]
    well_tracker_hc = ~np.isnan(d_d_f1f2_hh_cc)
    d_d_f1f2_cc_tt = d_f1f2[2,:]-d_f1f2[1,:]
    well_tracker_ct = ~np.isnan(d_d_f1f2_cc_tt)

    d_d_f1f2_hh_crcr = d_f1f2[3,:]-d_f1f2[0,:]
    well_tracker_hcr = ~np.isnan(d_d_f1f2_hh_crcr)
    d_d_f1f2_crcr_tt = d_f1f2[2,:]-d_f1f2[3,:]
    well_tracker_crt = ~np.isnan(d_d_f1f2_crcr_tt)


    
    idx_norm_f1_f2_face = np.isnan(norm_f1_f2_face)

    norm_f1_f2_face[well_tracker_ht] = (d_d_f1f2_hh_tt[well_tracker_ht]/d_ht_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_ht)

    well_tracker_hc_face = np.logical_and(idx_norm_f1_f2_face,well_tracker_hc)
    norm_f1_f2_face[well_tracker_hc_face] = (d_d_f1f2_hh_cc[well_tracker_hc_face]/d_hc_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_hc_face)

    well_tracker_ct_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_ct)
    norm_f1_f2_face[well_tracker_ct_face] = (d_d_f1f2_cc_tt[well_tracker_ct_face]/d_ct_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_ct_face)

    well_tracker_hcr_face = np.logical_and(idx_norm_f1_f2_face,well_tracker_hcr)
    norm_f1_f2_face[well_tracker_hcr_face] = (d_d_f1f2_hh_crcr[well_tracker_hcr_face]/d_hc_norm)

    well_tracker_crt_face = np.logical_and(idx_norm_f1_f2_face,well_tracker_crt)
    norm_f1_f2_face[well_tracker_crt_face] = (d_d_f1f2_crcr_tt[well_tracker_crt_face]/d_ct_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_crt_face)

    norm_f1_f2_face_wocr[:] = norm_f1_f2_face
    

    ############ Cross ###############
    
    d_f1f2_H1Cr2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,0,:,:]-loc_temp_ds[1,3,:,:]),axis=0))
    d_f1f2_C1Cr2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,1,:,:]-loc_temp_ds[1,3,:,:]),axis=0))
    d_f1f2_Cr2T1 = np.sqrt(np.sum(np.square(loc_temp_ds[0,2,:,:]-loc_temp_ds[1,3,:,:]),axis=0))
    d_f1f2_H2Cr1 = np.sqrt(np.sum(np.square(loc_temp_ds[1,0,:,:]-loc_temp_ds[0,3,:,:]),axis=0))
    d_f1f2_C2Cr1 = np.sqrt(np.sum(np.square(loc_temp_ds[1,1,:,:]-loc_temp_ds[0,3,:,:]),axis=0))
    d_f1f2_Cr1T2 = np.sqrt(np.sum(np.square(loc_temp_ds[1,2,:,:]-loc_temp_ds[0,3,:,:]),axis=0))
    
    d_f1f2_Cr1Cr2 = np.sqrt(np.sum(np.square(loc_temp_ds[0,3,:,:]-loc_temp_ds[1,3,:,:]),axis=0))

    d_d_f1_f2_h1cr2_cr2t1 = d_f1f2_Cr2T1-d_f1f2_H1Cr2
    well_tracker_ht1_cr2 = ~np.isnan(d_d_f1_f2_h1cr2_cr2t1)
    d_d_f1_f2_h1cr2_cr2c1 = d_f1f2_C1Cr2- d_f1f2_H1Cr2
    well_tracker_hc1_cr2 = ~np.isnan(d_d_f1_f2_h1cr2_cr2c1)
    d_d_f1_f2_c1cr2_cr2t1 = d_f1f2_Cr2T1-d_f1f2_C1Cr2
    well_tracker_ct1_cr2 = ~np.isnan(d_d_f1_f2_c1cr2_cr2t1)

    d_d_f1_f2_h2cr1_cr1t2 = d_f1f2_Cr1T2-d_f1f2_H2Cr1
    well_tracker_ht2_cr1 = ~np.isnan(d_d_f1_f2_h2cr1_cr1t2)
    d_d_f1_f2_h2cr1_cr1c2 = d_f1f2_C2Cr1-d_f1f2_H2Cr1
    well_tracker_hc2_cr1 = ~np.isnan(d_d_f1_f2_h2cr1_cr1c2)
    d_d_f1_f2_c2cr1_cr1t2 = d_f1f2_Cr1T2-d_f1f2_C2Cr1
    well_tracker_ct2_cr1 = ~np.isnan(d_d_f1_f2_c2cr1_cr1t2)
    
    well_tracker_cr1_cr2 = ~np.isnan(d_f1f2_Cr1Cr2)
    
    well_tracker_ht1_cr2_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_ht1_cr2)
    norm_f1_f2_face[well_tracker_ht1_cr2_face] = (d_d_f1_f2_h1cr2_cr2t1[well_tracker_ht1_cr2_face]/d_ht_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_ht1_cr2_face)

    well_tracker_hc1_cr2_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_hc1_cr2)
    norm_f1_f2_face[well_tracker_hc1_cr2_face] = (d_d_f1_f2_h1cr2_cr2c1[well_tracker_hc1_cr2_face]/d_hc_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_hc1_cr2_face)

    well_tracker_ct1_cr2_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_ct1_cr2)
    norm_f1_f2_face[well_tracker_ct1_cr2_face] = (d_d_f1_f2_c1cr2_cr2t1[well_tracker_ct1_cr2_face]/d_ct_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_ct1_cr2_face)

    well_tracker_ht2_cr1_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_ht2_cr1)
    norm_f1_f2_face[well_tracker_ht2_cr1_face] = (d_d_f1_f2_h2cr1_cr1t2[well_tracker_ht2_cr1_face]/d_ht_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_ht2_cr1_face)

    well_tracker_hc2_cr1_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_hc2_cr1)
    norm_f1_f2_face[well_tracker_hc2_cr1_face] = (d_d_f1_f2_h2cr1_cr1c2[well_tracker_hc2_cr1_face]/d_hc_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_hc2_cr1_face)

    well_tracker_ct2_cr1_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_ct2_cr1)
    norm_f1_f2_face[well_tracker_ct2_cr1_face] = (d_d_f1_f2_c2cr1_cr1t2[well_tracker_ct2_cr1_face]/d_ct_norm)
    idx_norm_f1_f2_face = np.logical_and(idx_norm_f1_f2_face, ~well_tracker_ct2_cr1_face)
    
    well_tracker_cr1_cr2_face = np.logical_and(idx_norm_f1_f2_face, well_tracker_cr1_cr2)
    norm_f1_f2_face[well_tracker_ct2_cr1_face] = 0
    
    return norm_f1_f2_face, norm_f1_f2_face_wocr
