from PIL import Image, ImageChops, ImageEnhance

# def get_imlist(path):
#     return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

def convert_to_ela_image(path, quality):
    im = Image.open(path).convert('RGB')
    resaved_filename = path.split('.')[0] + '.resaved.jpg'
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    # Prevent division by zero
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff 
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im

# Example usage for an original image
original_image_path = 'test_ela/Au_ani_10104.jpg'
original_image = Image.open(original_image_path)
original_image.show()

ela_image_original = convert_to_ela_image(original_image_path, 90)
ela_image_original.show()

# Example usage for a forged image
forged_image_path = 'test_ela/Tp_D_CRN_M_N_ani10104_ani00100_10093.tif'
forged_image = Image.open(forged_image_path)
forged_image.show()

ela_image_forged = convert_to_ela_image(forged_image_path, 90)
ela_image_forged.show()
