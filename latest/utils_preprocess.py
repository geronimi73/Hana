import torchvision.transforms as T
import numpy as np

from PIL import PngImagePlugin
# otherwise might lead to Decompressed Data Too Large for some images
LARGE_ENOUGH_NUMBER = 10
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
import fcntl
import os
import time

LOCK_FILE_PATH = 'lock_file.lock'

def acquire_lock():
    # Open the lock file in write mode
    lock_file = open(LOCK_FILE_PATH, 'w')

    # Try to acquire an exclusive lock on the file
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # print("Lock acquired.")
        return lock_file
    except IOError as e:
        # If the lock is already held by another process, wait and retry
        # print("Lock is held by another process. Waiting...")
        time.sleep(4)
        return acquire_lock()

def release_lock(lock_file):
    # Release the lock and close the file
    fcntl.flock(lock_file, fcntl.LOCK_UN)
    lock_file.close()
    # print("Lock released.")

    # Remove the lock file
    os.remove(LOCK_FILE_PATH)

## For example
ASPECT_RATIOS = [
    ("AR_1_to_1", 1),
    ("AR_4_to_3", 4/3),
    ("AR_3_to_4", 3/4),
]

def pad(img):
    w, h = img.size
    diff = abs(w-h)
    transform = T.Pad([0, diff//2] if w>h else [diff//2, 0], fill=get_average_color(img))
    return transform(img)

def get_average_color(img):
    img_array = np.array(img)
    return tuple(map(int, img_array.mean(axis=(0,1))))

# DC-AE compression factor=32
def closest_mult_of(num, denominator=32):
    return int(num//denominator) * denominator

def closest_ar_bucket(img, ARs=None):
    if ARs is None: 
        ARs = ASPECT_RATIOS
    w, h = img.size
    ar = w/h
    arbucket_diff = [abs(ar-ar_bucket) for ar_label, ar_bucket in ARs]
    arbucket_idx = arbucket_diff.index(min(arbucket_diff))

    return ARs[arbucket_idx]

def resize(img, resizeTo=None, debug=False, ARs=None):
    assert resizeTo is not None
    def noop(x): return x
    img = img.convert('RGB') if img.mode!="RGB" else img
    # smallest side to (resizeTo)px
    img = T.Resize(resizeTo, antialias=True)(img)
    # find closest aspect ratio bucket
    ar_name, ar = closest_ar_bucket(img, ARs=ARs)

    # Center crop 128x128, 128x?? or ??x128
    if ar == 1:
        # square
        if debug: print("Cropping square")
        img = T.CenterCrop(resizeTo)(img)
    elif ar>1:
        # landscape -> crop new_width x 128
        new_w = closest_mult_of(ar * resizeTo)
        if debug: print("Cropping landscape",new_w,"x",resizeTo)
        img = T.CenterCrop((resizeTo, new_w))(img)   # CenterCrop takes (h, w) !!
    else:
        # portrait -> crop 128 x new_height
        new_h = closest_mult_of(1/ar * resizeTo)
        if debug: print("Cropping portrait",resizeTo,"x",new_h)
        img = T.CenterCrop((new_h, resizeTo))(img)
            
    return ar_name, img
