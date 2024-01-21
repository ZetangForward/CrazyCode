from modelzipper.tutils import *

def main(rd):
    # load the data
    file_paths = []
    for i in range(8):
        file_paths.append(os.path.join(rd, f"inference_full_data_compress_1_snaps_{i}.pkl"))
    
    content = [auto_read_data(item) for item in file_paths]
    
    # merge content
    print_c(f"======= merge content =======", "magenta")
    merged_list = [item for sublist in content for item in sublist]

    # save the data
    print_c(f"======= save content =======", "magenta")
    save_path = os.path.join(rd, f"inference_full_data_compress_1_snaps_merged.pkl")
    b_t = time.time()
    auto_save_data(merged_list, save_path)
    print_c(f"save predictions to {save_path}, total time: {time.time() - b_t}", "magenta")

if __name__ == "__main__":
    fire.Fire(main)
    
    