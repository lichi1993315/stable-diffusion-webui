import io
import typing
from typing import Union, Optional, Tuple, Any

import numpy as np
from PIL.Image import Image as PILImage
from rembg import new_session
from rembg.bg import ReturnType, fix_image_orientation, post_process, alpha_matting_cutout, putalpha_cutout, \
    naive_cutout, get_concat_v_multi, apply_background_color
from rembg.sessions import BaseSession
from PIL import Image, ImageOps, ImageChops


def my_remove(
    data: Union[bytes, PILImage, np.ndarray],
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    session: typing.List[Optional[BaseSession]] = None,
    only_mask: bool = False,
    post_process_mask: bool = False,
    bgcolor: Optional[Tuple[int, int, int, int]] = None,
    *args: Optional[Any],
    **kwargs: Optional[Any]
) -> Union[bytes, PILImage, np.ndarray]:
    """
    Remove the background from an input image.

    This function takes in various parameters and returns a modified version of the input image with the background removed. The function can handle input data in the form of bytes, a PIL image, or a numpy array. The function first checks the type of the input data and converts it to a PIL image if necessary. It then fixes the orientation of the image and proceeds to perform background removal using the 'u2net' model. The result is a list of binary masks representing the foreground objects in the image. These masks are post-processed and combined to create a final cutout image. If a background color is provided, it is applied to the cutout image. The function returns the resulting cutout image in the format specified by the input 'return_type' parameter.

    Parameters:
        data (Union[bytes, PILImage, np.ndarray]): The input image data.
        alpha_matting (bool, optional): Flag indicating whether to use alpha matting. Defaults to False.
        alpha_matting_foreground_threshold (int, optional): Foreground threshold for alpha matting. Defaults to 240.
        alpha_matting_background_threshold (int, optional): Background threshold for alpha matting. Defaults to 10.
        alpha_matting_erode_size (int, optional): Erosion size for alpha matting. Defaults to 10.
        session (Optional[BaseSession], optional): A session object for the 'u2net' model. Defaults to None.
        only_mask (bool, optional): Flag indicating whether to return only the binary masks. Defaults to False.
        post_process_mask (bool, optional): Flag indicating whether to post-process the masks. Defaults to False.
        bgcolor (Optional[Tuple[int, int, int, int]], optional): Background color for the cutout image. Defaults to None.
        *args (Optional[Any]): Additional positional arguments.
        **kwargs (Optional[Any]): Additional keyword arguments.

    Returns:
        Union[bytes, PILImage, np.ndarray]: The cutout image with the background removed.
    """
    if isinstance(data, PILImage):
        return_type = ReturnType.PILLOW
        img = data
    elif isinstance(data, bytes):
        return_type = ReturnType.BYTES
        img = Image.open(io.BytesIO(data))
    elif isinstance(data, np.ndarray):
        return_type = ReturnType.NDARRAY
        img = Image.fromarray(data)
    else:
        raise ValueError("Input type {} is not supported.".format(type(data)))

    putalpha = kwargs.pop("putalpha", False)

    # Fix image orientation
    img = fix_image_orientation(img)

    if session is None:
        session = new_session("u2net", *args, **kwargs)

    mask1 = session[0].predict(img, *args, **kwargs)[0]
    mask2 = session[1].predict(img, *args, **kwargs)[0]

    combined_mask = ImageChops.add(mask1, mask2)
    masks = [combined_mask]

    cutouts = []

    for mask in masks:
        if post_process_mask:
            mask = Image.fromarray(post_process(np.array(mask)))

        if only_mask:
            cutout = mask

        elif alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                if putalpha:
                    cutout = putalpha_cutout(img, mask)
                else:
                    cutout = naive_cutout(img, mask)
        else:
            if putalpha:
                cutout = putalpha_cutout(img, mask)
            else:
                cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    cutout = img
    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)

    if bgcolor is not None and not only_mask:
        cutout = apply_background_color(cutout, bgcolor)

    if ReturnType.PILLOW == return_type:
        return cutout

    if ReturnType.NDARRAY == return_type:
        return np.asarray(cutout)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    bio.seek(0)

    return bio.read()
