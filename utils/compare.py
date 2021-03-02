import os, sys
from os import listdir
from os.path import isfile, join

def get_file_list(path):
    ''' given absolute path, return list of files in directory '''
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    return files

def and_files(path1, path2):
    

