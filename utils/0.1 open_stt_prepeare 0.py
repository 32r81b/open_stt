import pandas as pd
import wave
import contextlib
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import tqdm
import time
import librosa
import soundfile as sf

# path = 'C://data/'
path = '/data/'

datasets = pd.DataFrame(columns=['name', 'wav_filename', 'wav_filesize', 'transcript', 'duration'])
dataset_list = pd.DataFrame(columns=['name', 'pos_wav_filename', 'pos_transcript_file'])
exclude = pd.read_csv('public_exclude_file_v5.csv')


# Datasets defenition:
#   name - file name of dataset
#   pos_wav_filename - number of column with pos_wav_filename
#   pos_wav_filename - number of column with pos_transcript_filename
dataset_list.loc[0] = ['public_youtube700.csv', 0, 1]
# dataset_list.loc[1] = ['asr_calls_2_val.csv', 0, 1]
# dataset_list.loc[2] = ['public_youtube1120_hq.csv', 0, 1]
# dataset_list.loc[3] = ['asr_public_phone_calls_1.csv', 0, 1]
# dataset_list.loc[4] = ['asr_public_phone_calls_2.csv', 0, 1]
# dataset_list.loc[5] = ['private_buriy_audiobooks_2.csv', 0, 1]


wav_notfound_name = []
txt_notfound_name = []

sr = 8000

for index, dataset in dataset_list.iterrows():
    duration = []
    size = []
    transcript = []
    length = []

    dataset_name = dataset['name'][:-4]

    wav_notfound = 0
    txt_notfound = 0
    exclude_num = 0
    wav_filename_8k = []

    print('Loading: ' + dataset['name'])

    dataset_rows = pd.read_csv(str(path + 'ru_open_stt/' + dataset['name']), header=None)
    exclude_ds = exclude[exclude.dataset == dataset_name]
    merged = pd.merge(left=dataset_rows, right=exclude_ds, how='left', left_on=0, right_on='wav')
    dataset_rows = merged[dataset_rows.columns][merged['wav'].isna()]

    print('Number of saved rows: ' + str(dataset_rows.shape[0]))
    print('Number of excluded rows: ' + str(merged[merged['wav'].notna()].shape[0]))

    wav_filename = dataset_rows.loc[:, dataset['pos_wav_filename']]
    for filname in wav_filename:
        wav_filename_8k.append(str(filname[0:19] + '_8k' + filname[19:]))

    iter_len = len(wav_filename)

    print('Start processing WAV files...')
    time.sleep(2)

    for i in tqdm.tqdm(range(iter_len)):
        wav_file = wav_filename[wav_filename.index[i]]
        wav_file_8k = wav_filename_8k[i]
        fname = str(path + wav_file)
        fname_8k = str(path + wav_file_8k)
        if Path(fname).is_file():
            with contextlib.closing(wave.open(fname, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                value = frames / float(rate)
                # To prevent error 'Input signal is too small to resample from 16000->8000' exclude short samples
                if value > 2:
                    duration.append(value)
                    filestat = os.stat(fname)
                    size.append(filestat.st_size)
                    y, s = librosa.load(fname, sr=sr)
                    Path(os.path.dirname(fname_8k)).mkdir(parents=True, exist_ok=True)
                    sf.write(fname_8k, y, sr, subtype='PCM_16')

                else:
                    duration.append(0)
                    size.append(0)
        else:
            wav_notfound = wav_notfound + 1
            duration.append(0)
            size.append(0)
            wav_notfound_name.append(fname)

    time.sleep(2)
    print(str(wav_notfound) + ' WAV files not found')
    transcript_filename = dataset_rows.loc[:, dataset['pos_transcript_file']]

    print('Start processing TXT files...')
    time.sleep(2)

    for i in tqdm.tqdm(range(iter_len)):
        transcript_file = transcript_filename[transcript_filename.index[i]]
        fname = str(path + transcript_file)
        if Path(fname).is_file():
            f = open(fname, 'r', encoding='utf-8')
            line = f.readline()[:-1]
            transcript.append(line.lower())
            length.append(len(str(line.lower())))
        else:
            txt_notfound = txt_notfound + 1
            transcript.append('missed')
            length.append(0)
            txt_notfound_name.append(fname)

    time.sleep(2)

    print(str(txt_notfound) + ' TXT files not found')

    name = pd.Series([dataset['name']] * len(dataset_rows))

    datasets_tmp = pd.DataFrame({'name': name, 'wav_filename': wav_filename_8k, 'wav_filesize': size, 'length': length,
                                 'transcript': transcript, 'duration': duration})
    datasets = datasets.append(datasets_tmp, sort=False)
    print('Dataset duration (hour): ' + str(sum(duration)/3600))
    print('')


datasets = datasets[(datasets.duration > 3) & (datasets.duration < 10) & (datasets.length > 10) & (datasets.length < 120)]



# train, dev = train_test_split(datasets[(datasets.name == 'asr_public_phone_calls_1.csv') | (datasets.name == 'asr_public_phone_calls_2.csv')], test_size=0.3, random_state=42)
# test = datasets[datasets.name == 'asr_calls_2_val.csv']

train, dev = train_test_split(datasets, test_size=0.2, random_state=42)


train.to_csv(path + '0clean_train.csv')
dev.to_csv(path + '0clean_dev.csv')
# datasets.to_csv(path + 'transcript.csv', columns=['transcript'], index = False, header = False)


# print(wav_notfound_name)
# print()
# print(txt_notfound_name)
