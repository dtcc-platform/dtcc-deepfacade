import os
 
def find_files_in_folder(folder, extension):
    if os.path.exists(folder):
        paths= []
        for file in os.listdir(folder):
            if file.endswith(extension):
                paths.append(os.path.join(folder , file))

        return paths
    else:
        raise Exception("path does not exist -> "+ folder)