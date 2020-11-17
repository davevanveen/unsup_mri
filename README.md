# this codebase is based on the original run on pytorch==1.5, but has been "minimally" modified to run on pytorch==1.7

- main difference between pt==1.5 and pt==1.7 is how the fft is processed - i.e. pt==1.5 requires shape (nc,x,y,2) w real/complex channels separate in last dimension, while pt==1.7 requires shape (nc,x,y) of complex dtype
- in order to revert back, just need to update how fft is performed
	- ifft is fine b/c using a deprecated function, but fft must be manually edited
- see `utils.transfrom.py` and comment out extra dim reshape code in fft_2d(), i.e. get it to mirror ifft_2d(). also need to call torch.fft() instead of torch.fft.fftn()
- maybe there are other changes i'm missing?


### see `20201112_eval_fastmri_old_v_new.ipynb` in dev branch for further experimental details
