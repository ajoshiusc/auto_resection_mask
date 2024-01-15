
from make_wm_lesion import make_wm_lesion


#IXI062-Guys-0740
#core_modify_factor = 1 means replace the intensity in the lesion core by a constant value = core_modify_factor * (average of intensity in the initial lesion core)
# > 1 will lead to hyper intensitues and < 1 will lead to hypointense lesion

#make_wm_lesion('IXI452-HH-2213',core_modify_factor=2.5,out_dir = '/scratch1/ajoshi/temp_dir')
make_wm_lesion('IXI062-Guys-0740',core_modify_factor=2.5,out_dir = '/home/ajoshi/Desktop/temp_dir')
