import os
import wget
import tarfile

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
    
if __name__=="__main__":
    download_VOC_2012()
