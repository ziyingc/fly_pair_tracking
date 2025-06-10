exp_target = 'R72'
gtype_name = ['R72A10xTRPA1','R72A10xW1118','W1118xTRPA1']

h5_dir = '../../FlyBehaviorSet/DLC_tracking/'


sex_name = ['female','male']
output_fileloc = '../../FlyBehaviorSet/'
OG_recording_folder      = 'original_recording_for_tracking/'
converted_folder         = 'Converted/'
cropped_recording_folder = 'cropped_recording_for_tracking'  



dir_MCAM = '../DatastreamLine_MCAM_Tracking/'
scorer_floor  = 'DLC_resnet152_floor_topview_12label_1uniqueJan4shuffle1_100000'
scorer_wall   = 'DLC_resnet152_wall_sideview_7labelsJan1shuffle1_100000'
scorer_cross  = 'DLC_resnet152_wall_cross_section_view_8labelFeb19shuffle1_100000'


frame_n = 126000
frame_rate = 60

color_BP = [(0,0,255),(0,255,0),(255,0,0)]
arena_parameters = {
    'well_radius': 0.0084,  # meters
    'food_cup_radius': 0.0035,  # meters
}

experiments_index  = np.load('experiments_index.npy')
fileloc_set        = np.load('experiments_directory.npy')
num_file = len(fileloc_set)
videofile_set      = np.load('experiments_well_setup.npy')
num_well = len(videofile_set)
experiments_mscore = np.load('experiments_mscore.npy',allow_pickle='TRUE').item()
crop_h = np.load('crop_h.npy')