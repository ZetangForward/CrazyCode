from modelzipper.tutils import *


def analysis_cov1d_compress(fpath):
    file_names = auto_read_dir(fpath, ".pkl")

    for file_name in file_names:
        layer_idx = int(os.path.basename(file_name).split("-")[0].split(".")[0])
        hidden_state = auto_read_data(file_name)
       
        
        break
    
