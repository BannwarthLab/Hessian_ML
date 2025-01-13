
def list_to_txt(list, file):
    with open(file, "w+") as f:
        for d in list:
            f.write(d)
            f.write("\n")
    f.close()
