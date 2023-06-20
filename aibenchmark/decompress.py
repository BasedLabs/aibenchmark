import py7zr

from aibenchmark.exceptions import NotSupportableDecompressionFileFormat


def decompress_7z(file_path: str, destination_folder_path: str):
    '''
    Decompress from 7z archive
    :param destination_folder_path: output dir for file
    :param file_path: 7z file path
    :return: None
    '''
    if not file_path.endswith('.7z'):
        raise NotSupportableDecompressionFileFormat()

    with py7zr.SevenZipFile(file_path, mode='r') as z:
        z.extractall(destination_folder_path)
