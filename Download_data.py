import os
import wget
import tarfile
import zipfile


def download_VOC_2012():
    '''
    Downloads VOC_2012 dataset for object detection
    :return:
    '''
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    folder = './Pascal_VOC_2012'
    if os.path.exists(folder):
        print('Pascal Voc 2012 was already downloaded')
    else:
        os.mkdir(folder)
        wget.download(url, folder + '/')
        print('Download is completed')

    file_list = os.listdir(folder)
    if file_list == []:
        raise RuntimeError('Download might be failed.')

    for i in range(len(file_list)):
        element = file_list[i]
        if 'VOCtrainval' in element:
            file_name = element

            with tarfile.TarFile(folder + '/' + file_name, 'r') as tf:
                tf.extractall(folder + '/')
                print('Voc 2012 is unzip')
                tf.close()
    

def download_COCO_2017_Val():
    url = 'http://images.cocodataset.org/zips/val2017.zip'
    folder = './MS_COCO_2017'

    os.mkdir(folder)
    wget.download(url, folder + '/')
    print('Download is completed')

    file_list = os.listdir(folder)
    if file_list == []:
        raise RuntimeError('Download might be failed.')

    for i in range(len(file_list)):
        element = file_list[i]
        if '.zip' in element:
            file_name = element
            with zipfile.ZipFile(folder + '/' + file_name, 'r') as tf:
                tf.extractall(folder + '/')
                print('COCO 2017 val is unzip')
                tf.close()

                
def download_COCO_2017_anno():
    url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    folder = './MS_COCO_2017_Anno'
    if os.path.exists(folder):
        print('MS COCO 2017_Anno was already downloaded')
    else:
        os.mkdir(folder)
        wget.download(url, folder + '/')
        print('Download is completed')
    file_list = os.listdir(folder)
    if file_list == []:
        raise RuntimeError('Download might be failed.')

    for i in range(len(file_list)):
        element = file_list[i]
        if '.zip' in element:
            file_name = element
            with zipfile.ZipFile(folder + '/' + file_name, 'r') as tf:
                tf.extractall(folder + '/')
                print('COCO 2017_anno is unzip')
                tf.close()


if __name__ == "__main__":
    # download_VOC_2012()
    download_COCO_2017_Val()
    download_COCO_2017_anno()
