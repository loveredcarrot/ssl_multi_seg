import os
import sys


def generator(root_path, save_path, txt_name, image_dir_name, mask_dir_name):
    image_path = os.path.join(root_path, image_dir_name)

    mask_path = os.path.join(root_path, mask_dir_name)

    nameList = os.listdir(image_path)

    f = open(os.path.join(save_path, txt_name + ".txt"), 'a+')

    for name in nameList:
        name_note, _ = os.path.splitext(name)

        i_name = os.path.join(image_path, name)
        m_name = os.path.join(mask_path, name_note + '.png')
        f.writelines([i_name, ',', m_name, '\n'])

    f.close()
    print('success output img_mask txt')


if __name__ == "__main__":
    # if len(sys.argv) != 6:
    #     print("please input root_path, save_path, txt_name, image_dir_name, mask_dir_name")
    #     exit(1)
    #
    # root_path = sys.argv[1]
    # save_path = sys.argv[2]
    # txt_name = sys.argv[3]
    # image_dir_name = sys.argv[4]
    # mask_dir_name = sys.argv[5]
    root_path = r'D:\mypythonproject\ssl_seg\data\test'
    save_path = r'D:\mypythonproject\ssl_seg\data\test'
    txt_name = r'train'
    image_dir_name = 'image'
    mask_dir_name = 'label'
    generator(root_path, save_path, txt_name, image_dir_name, mask_dir_name)
