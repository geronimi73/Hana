from PIL import PngImagePlugin, Image
# otherwise might lead to Decompressed Data Too Large for some images
LARGE_ENOUGH_NUMBER = 10
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = None

from io import BytesIO
import fcntl
import os, time, numpy as np, requests, torchvision.transforms as T
import aiohttp, asyncio


async def download_image(session, url, timeout=300):
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                print(f"Failed to download {url}: HTTP status {response.status}")
                return None
                
            content = await response.read()
            try:
                return Image.open(BytesIO(content))
            except Exception as e:
                print(f"Failed to process image from {url}: {str(e)}")
                return None
                
    except asyncio.TimeoutError:
        print(f"Timeout while downloading {url}")
        return None
    except aiohttp.ClientError as e:
        print(f"Client error while downloading {url}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error while downloading {url}: {str(e)}")
        return None

async def download_all_images(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def url_to_image(url):
    # Send a HTTP request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Open the image using PIL
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return None

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

def resizeToClosestMultOf(img, resizeTo=None, denominator=32, maxDim=None):
    """
    Resize an image to have dimensions that are multiples of a specific value. Intended for subsequent processing by DC-AE.
    
    This function performs a two-step operation:
    1. Resizes the smallest dimension of the input image to the target size
    2. Center crops the image so both dimensions are multiples of the denominator
    
    Parameters:
    -----------
    img : PIL.Image
        The input image to be processed.
    resizeTo : int, required
        The target size for the smallest dimension of the image in pixels.
    denominator : int, default=32
        The value of which both final dimensions should be multiples.
        
    Returns:
    --------
    (newWidth, newHeight)
        A tuple of the img dimensions after resizing (and cropping)

    PIL.Image
        The resized and cropped image with dimensions that are multiples
        of the specified denominator.
            
    """
    img = img.convert('RGB') if img.mode!="RGB" else img

    # First, resize smallest side to target
    img = T.Resize(resizeTo, antialias=True)(img)

    # Second, figure out which area to center crop, closest multiple of 32
    w, h = img.size
    newW, newH = closest_mult_of(w, denominator), closest_mult_of(h, denominator)

    # Third, restrict maxDim if needed
    if maxDim:
        newW, newH = min(newW, maxDim), min(newH, maxDim)

    return T.CenterCrop((newH, newW))(img)

def resizeToARBucket(img, resizeTo=None, debug=False, ARs=None):
    """
    Resize and crop an image to fit the closest aspect ratio bucket.
    
    This function takes an input image, resizes it so that its smallest dimension
    equals 'resizeTo', and then applies a center crop to match the closest predefined
    aspect ratio bucket.
    
    Parameters:
    -----------
    img : PIL.Image
        The input image to be processed.
    resizeTo : int, required
        The target size for the smallest dimension of the image in pixels.
    debug : bool, default=False
        If True, prints debugging information about cropping operations.
    ARs : dict, optional
        A dictionary of aspect ratio buckets. If None, uses default buckets
        defined in closest_ar_bucket().
        
    Returns:
    --------
    tuple
        A tuple containing:
        - ar_name (str): The name of the chosen aspect ratio bucket
        - img (PIL.Image): The resized and cropped image
        
    Notes:
    ------
    - Converts input image to RGB mode if it's not already
    - For square images (AR=1), crops to resizeTo×resizeTo
    - For landscape images (AR>1), crops to resizeTo×(AR*resizeTo)
    - For portrait images (AR<1), crops to (resizeTo/AR)×resizeTo
    - Dimensions are adjusted to be a multiple of 8 using closest_mult_of()
    - CenterCrop expects dimensions in (height, width) order
    """

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
