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
# Create the start object, create a directory_map, and make a frame from the directory_map with depth of 1 consisting only of folders. This means the names column of the frame consists of the names of the folders that reside at the given path,
lsobject = ls.start(directory=path)
lsobject.map_directory()
lsobject.map_to_frame(depth=1, kind="folders")
# %%

# %% [markdown]
# Scan through the names of the folders collected in the frame and create zip files of the contents,
[
    shutil.make_archive(os.path.join(path, name), "zip", os.path.join(path, name))
    for name in lsobject.frame.name
]
# %%
