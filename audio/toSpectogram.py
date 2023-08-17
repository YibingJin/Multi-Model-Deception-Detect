#matplotlib inline
import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
audiopath='./audioclip'
dir=os.listdir(audiopath)
for a in dir:
    if os.path.isfile(audiopath + "/" + a):
        path=audiopath + "/" + a
        fileName = os.path.splitext(a)[0]
        y, sr = librosa.load(path, sr=16000)
        
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        #plt.figure(figsize=[2.24,2.24],dpi=100)
        librosa.display.specshow(mel_spect, fmax=sr)
        filename=fileName+'.png'
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        #plt.title(filename)
        plt.savefig(filename,bbox_inches = 'tight')
        plt.show()

        print(filename)







mel_savepath='./processedaudio'

# audio_path是歌曲的保存路径，需要是load方法可以读取的歌曲文件格式



