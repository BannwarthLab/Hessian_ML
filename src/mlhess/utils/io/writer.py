import os


def truncate_file(file: str):
    """Truncate a file.

    Args:
        file (str): File name.
    """
    if os.path.isfile(file):
        with open(file, "wb+") as f:
            f.truncate(0)
        f.close()


def list_to_txt(list: list, file: str):
    """Write a list to a txt file.

    Args:
        list (list): List to convert.
        file (str): Name of the file.
    """
    with open(file, "w+") as f:
        for d in list:
            f.write(d)
            f.write("\n")
    f.close()
