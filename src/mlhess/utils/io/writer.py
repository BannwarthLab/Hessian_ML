import os


def truncate_file(file: str):
    """Truncate a file

    :param file: file name
    :type file: str
    """
    if os.path.isfile(file):
        with open(file, "wb+") as f:
            f.truncate(0)
        f.close()


def list_to_txt(list: list, file: str):
    """Write a list to a txt file.

    :param list: list to converted
    :type list: list
    :param file: name of the file
    :type file: str
    """
    with open(file, "w+") as f:
        for d in list:
            f.write(d)
            f.write("\n")
    f.close()
