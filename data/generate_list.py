import os

def genreate_list(root, split):

    list_path = os.path.join(root, split+'.txt')
    with open(list_path, 'w') as f:
        for img_name in os.listdir(os.path.join(root)):
            f.write(img_name + '\n')

if __name__ == "__main__":
    root = r'E:\cddataset\mmcd\Second_my\val\im1'
    genreate_list(root, 'val')