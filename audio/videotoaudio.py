from moviepy.editor import *
import os
##该文件作用：提取所有视频文件的音频

##第一步、加载文件，获取文件路径以及标签
train_path = "D:\PyCharm 2021.3.2\openface\\visual\Clips"
allpath = []
lllables = []


def get_lableandwav(path, dir):
    dirs = os.listdir(path)
    for a in dirs:
        print(a)
        print(os.path.isfile(path + "/" + a))
        if os.path.isfile(path + "/" + a):
            video = VideoFileClip(path + "/" + a)
            audio = video.audio
            file_name=a+'.wav'
            audio.write_audiofile(file_name)
            if dir != "":
                lllables.append(dir)
        else:
            get_lableandwav(str(path) + "/" + str(a), a)
        ##循环遍历这个文件夹

    return allpath, lllables


##第一步、加载文件，获取文件路径以及标签
[allpath, lllables] = get_lableandwav(train_path, "")
print('allpath:',allpath)
print("----------")
print(lllables)
# video = VideoFileClip('E:\deception detection\deleted videos\deceptive\\trial_lie_045.mp4')
# audio = video.audio
# audio.write_audiofile('test.wav')
