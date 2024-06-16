import torchvision

totensor = torchvision.transforms.ToTensor()

def transform_augment_cd(img, min_max=(0, 1)):
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img