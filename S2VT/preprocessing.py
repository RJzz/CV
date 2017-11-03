# -*- coding:utf-8 -*-
import pandas as pd
import cv2
import imageio
import skimage
import pylab

VIDEO_INFO_PATH = '/home/rjzz/dataset/video/MSVD/MSR Video Description Corpus.csv'
VIDEO_OBJECT_PATH = '/home/rjzz/dataset/video/MSVD/Video/'


class Video:
    def __init__(self, id, start, end):
        self.id = str(id)
        self.start = int(start)
        self.end = int(end)


# 裁剪视频
def cut_video(video_object):
    full_path = VIDEO_OBJECT_PATH + video_object.__getattribute__('id') + '.mp4'
    video_capture = cv2.VideoCapture(full_path)
    # if video_capture
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    return video_capture


if __name__ == '__main__':
    video_info = pd.read_csv(VIDEO_INFO_PATH)

    # 文中说明只是用English标注来训练
    video_info = video_info[video_info['Language'] == 'English']

    # 去掉重复的
    video_info = video_info[['VideoID', 'Start', 'End']]
    video_object = video_info.drop_duplicates()

    img = cv2.imread('./1.png')
    cv2.imshow('test', img)
    # video = cv2.VideoCapture('./mv89psg6zh4.avi')

    # fps = video.get(cv2.CAP_PROP_FPS)

    video = imageio.get_reader('./mv89psg6zh4.avi')
    # fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    frame_list = []
    for i, img in enumerate(video):
        # frame_list.append(img)
        frame_count += 1
        # fig.suptitle('image # {}'.format(i), fontsize=20)
        # pylab.imshow(img)
        opencv_img = skimage.img_as_ubyte(img, True)
        # print(opencv_img.shape)
    # 遍历处理视频\


    print('hehe')
    # for iter in video_object.index:
    #     name = video_object.loc[iter, 'VideoID']
    #     start = video_object.loc[iter, 'Start']
    #     end = video_object.loc[iter, 'End']
    #     video = Video(name, start, end)
    #     cut_video(video)
    print(video_object.index)
