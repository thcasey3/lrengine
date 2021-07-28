# sphinx_gallery_thumbnail_path = '_static/logo-white.png'
# %% [markdown]
"""
zip folders
===========

This example demonstrates zipping the subfolders of a parent directory.

"""
# %%

# %% [markdown]
# Import lsframe, import shutil for creating the zip files, and import os for defining paths,
import lsframe as ls
import shutil
import os

# %%

# %% [markdown]
# Set the path to a parent directory full of folders you would like to zip,
path = "../path/to/parent/directory"
# %%


# %% [markdown]
# Create the start object, create a directory_map, and make a frame from the directory_map with depth of 1 and only consisting of folders. The means the names of the frame are a list of folders that are in the given path,
lsobject = ls.start(directory=path)
lsobject.map_directory()
lsobject.map_to_frame(depth=1, kind="folders")
# %%

# %% [markdown]
# Loop over the names of the frame and create zip files from the corresponding directories,
[
    shutil.make_archive(os.path.join(path, name), "zip", os.path.join(path, name))
    for name in lsobject.frame.name
]
# %%
