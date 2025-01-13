import os 

def rd_txt_file(file):
    with open(f"{file}", "rb") as f:
        filenames = f.read().splitlines()
    f.close()

    for i in range(len(filenames)):
        filenames[i] = filenames[i].decode("ascii")

    return filenames

def truncate_file(file):
    if os.path.isfile(file):
        with open(file, "wb+") as f:
            f.truncate(0)
        f.close()

