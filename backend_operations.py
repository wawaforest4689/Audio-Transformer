from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from pydub import AudioSegment
# from pydub.silence import detect_nonsilent
import numpy as np
import os
from scipy.fftpack import fft
from scipy.fftpack import ifft
import scipy.signal as signal
from pydub.playback import play
import matplotlib.pyplot as plt
from practice13 import pseudocolor_enhance
import cv2


def play_music(input_path):
    os.system('chcp 65001')
    command = f'wmplayer.exe {input_path} /play'
    print(f"command:{command}\n")
    result = os.system(command)
    # command=[r"wmplayer",input_path,' /play']
    # result=subprocess.run(command,shell=True,text=True)
    # print(result)
    # print(result.returncode)
    # print(result.stdout)


def comb_music(input_path1, input_path2, output_path):
    format1 = input_path1[-3:]
    format2 = input_path2[-3:]
    filename1 = os.path.basename(input_path1)
    filename2 = os.path.basename(input_path2)

    music1, music2 = None, None
    if format1 == 'wav':
        music1 = AudioSegment.from_wav(input_path1)
    elif format1 == 'mp3':
        music1 = AudioSegment.from_mp3(input_path1)
    else:
        real_format = filename1[max(-8, -len(filename1)):]
        real_format = real_format.strip(".")[1]
        print(f"File1: Unsupported audio file type for {real_format}")

    if format2 == 'wav':
        music2 = AudioSegment.from_wav(input_path2)
    elif format2 == 'mp3':
        music2 = AudioSegment.from_mp3(input_path2)
    else:
        real_format = filename2[max(-8, -len(filename2)):]
        real_format = real_format.strip(".")[1]
        print(f"File2: Unsupported audio file type for {real_format}")

    fs1, fw1, ch1 = music1.frame_rate, music1.frame_width, music1.channels
    fs2, fw2, ch2 = music2.frame_rate, music2.frame_width, music2.channels

    # 基础信息校准
    if fs1 != fs2:
        print(f"Error:Sampling rate unequal.{fs1, fs2} Hz")
        return
    if fw1 != fw2:
        print(f"Error:Frame width unequal.{fw1, fw2} bytes")
        return
    if ch1 != ch2:
        print(f"Error:Channels unequal.{ch1, ch2} channels")
        return

    data = np.asarray(music1.get_array_of_samples(), float).reshape(-1, ch1)
    data2 = np.asarray(music2.get_array_of_samples(), float).reshape(-1, ch2)

    comb_data = data + data2

    # 初始格式化为WAV文件格式
    if fw1 / ch1 == 2:
        comb_data = np.asarray(comb_data / np.max(abs(comb_data)) * 2 ** 15 * 0.7, np.int16)
    else:
        comb_data = np.asarray(comb_data / np.max(abs(comb_data)) * 2 ** 31 * 0.7, np.int32)

    comb_music = AudioSegment(data=comb_data.tobytes(), frame_rate=fs1, sample_width=int(fw1 / ch1), channels=ch1)
    # 导出到指定目录（必须指定）
    filename = filename1[:-4] + '__' + filename2[:-4] + '__combine.mp3'
    comb_music.export(output_path + filename, 'mp3')

    return output_path + filename


def separate_vocals(input_path, output_dir='output', model_type='2stems',
                    model_path='d:/wf200/Documents/mypython/AI_models'):
    """
    分离音频中的人声和伴奏（支持自定义模型路径）
    :param input_path: 输入音频文件路径
    :param output_dir: 输出目录
    :param model_type: 使用的模型 (可选: '2stems', '4stems', '5stems')
    :param model_path: 自定义模型根目录路径
    """

    # 获取歌曲名称（创建目录用）,drop ".mp3"/".wav"
    file_name = os.path.basename(input_path)[:-4]
    output_dir = output_dir + '/' + file_name

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    #   切换工作目录到模型文件存放目录
    os.chdir(model_path)

    try:
        # 创建 AudioAdapter
        adapter = AudioAdapter.default()
        mixed_music, sample_rate = adapter.load(input_path)

        # 初始化分离器（加载模型）
        separator = Separator(f'spleeter:{model_type}')
        # 验证模型是否加载成功
        # 尝试获取模型内部属性
        print(f"模型类型: {separator._params['model']}")
        print(f"采样率: {separator._params['sample_rate']}")
        print(f"帧大小: {separator._params['frame_length']}")

        # 检查模型是否已加载
        if hasattr(separator, '_prediction_generator'):
            print("✅ 模型加载成功")
        else:
            print("❌ 模型未加载 - 可能是路径问题")
        # 执行分离操作
        vocal = separator.separate(mixed_music)['vocals']
        bgmusic = mixed_music - vocal
        adapter.save(output_dir + r'\vocals.wav', vocal, sample_rate)
        adapter.save(output_dir + r'\bgmusic.wav', bgmusic, sample_rate)

        # separator.separate_to_file(input_path,output_dir,codec="wav",filename_format='{instrument}.{codec}',synchronous=True)

        print(f"分离成功！结果保存在: {os.path.abspath(output_dir)}")
        return True

    except Exception as e:
        print(f"分离失败: {str(e)}")
        return False


def separate_vocals_group(input_path, model_type='2stems', model_path='d:/wf200/Documents/mypython/AI_models'):
    """
    分离音频中的人声和伴奏（支持自定义模型路径）
    :param input_path: 输入音频文件路径
    :param output_dir: 输出目录（同输入音频文件目录）
    :param model_type: 使用的模型 (可选: '2stems', '4stems', '5stems')
    :param model_path: 自定义模型根目录路径
    """

    #   切换工作目录到模型文件存放目录
    os.chdir(model_path)

    try:
        # 创建 AudioAdapter
        adapter = AudioAdapter.default()
        # 初始化分离器（加载模型）
        separator = Separator(f'spleeter:{model_type}')
        # 验证模型是否加载成功
        # 尝试获取模型内部属性
        print(f"模型类型: {separator._params['model']}")
        print(f"采样率: {separator._params['sample_rate']}")
        print(f"帧大小: {separator._params['frame_length']}")

        # 检查模型是否已加载
        if hasattr(separator, '_prediction_generator'):
            print("✅ 模型加载成功")
        else:
            print("❌ 模型未加载 - 可能是路径问题")

        root, dir, files = os.walk(input_path)
        for file in files:
            try:
                input_path = input_path + '/' + file
                print(input_path)
                mixed_music, sample_rate = adapter.load(input_path)

                # 获取歌曲名称（创建目录用）,drop ".mp3"/".wav"
                # 支持同目录导出
                base_l = len(os.path.basename(input_path))
                filename = os.path.basename(input_path)[:-4]
                dir = input_path[:-base_l]
                print(f'dir:{dir}\nfilename:{filename}\n')

                # 执行分离操作，直接返回值为字典，提取人声部分
                vocal = separator.separate(mixed_music)['vocals']
                # 对于2stems，4stems，5stems都只做人声-伴奏分离处理
                bgmusic = mixed_music - vocal
                adapter.save(dir + filename + '_vocals.mp3', vocal, sample_rate)
                adapter.save(dir + filename + '_bgmusic.mp3', bgmusic, sample_rate)
            except Exception as e:
                print(f'加载音频文件失败：{str(e)}')

        # separator.separate_to_file(input_path,output_dir,codec="wav",filename_format='{instrument}.{codec}',synchronous=True)

        print(f"分离成功！结果保存在: {dir}")
        return dir + filename + '_vocals.mp3', dir + filename + '_bgmusic.mp3'

    except Exception as e:
        print(f"分离失败: {str(e)}")
        return False


"""
min_time,step以毫秒为单位
v_thres以dB分贝为单位（正值），用于动态方法的静音区间检测
sound不是AudioSegment对象，是np array
"""


def remove_small_volumes(sound, min_time, v_thres, step, f_r):
    step = int(step * f_r / 1000)
    min_time = int(min_time * f_r / 1000)
    # 改进1：整段拼接避免爆音
    # 改进2：动态阈值判断小音量区间，消除大段静音区间对于音量方均根的拉低效应
    max_volume = np.max(abs(sound))
    # 区间包含头包含尾
    silent = []
    # 不考虑最后一部分时段
    for sindex in range(0, len(sound) - min_time, step):
        # rms=np.sqrt(np.sum([np.square(sound[sindex+i].rms) for i in range(min_time)])/min_time)
        # 这里使用先查找小音量（静音）区间的办法，然后做并集-补集操作
        rms = (np.sum(np.square(sound[sindex:sindex + min_time])) / min_time) ** 0.5
        if rms <= max_volume / 10 ** (abs(v_thres) / 20):
            silent.append([sindex, sindex + min_time - 1])
    i = 0
    while i < len(silent) - 1:
        if silent[i][1] >= silent[i + 1][0] - 1:
            silent[i][1] = silent[i + 1][1]
            del silent[i + 1]
        else:
            i = i + 1

    non_silent = [(0, len(sound) - 1)]
    nons_sound = sound
    if len(silent) > 0:
        if silent[0][0] > 0:
            non_silent.append((0, silent[0][0] - 1))

        for i in range(len(silent) - 1):
            non_silent.append((silent[i][1] + 1, silent[i + 1][0] - 1))

        if silent[-1][1] < len(sound) - 1:
            non_silent.append((silent[-1][1] + 1, len(sound) - 1))

        nons_sound = np.zeros((1, len(sound[0])))
        for mark in non_silent:
            nons_sound = np.concatenate((nons_sound, sound[mark[0]:mark[1] + 1]), axis=0)
        nons_sound = nons_sound[1:]

    return nons_sound, silent


def remove_silence_ui(input_path, min_time=200, v_thres=40, step=2):
    format = input_path[-3:]
    music = None
    if format == 'wav':
        music = AudioSegment.from_wav(input_path)
    elif format == 'mp3':
        music = AudioSegment.from_mp3(input_path)
    else:
        print("Unsupported audio file type!")
        return
    fs, fw, channels = music.frame_rate, music.frame_width, music.channels
    data = np.asarray(music.get_array_of_samples()).reshape(-1, channels)
    nons_sound, _ = remove_small_volumes(data, min_time, v_thres, step, fs)

    # 同目录导出(不需要进行整数化和范围限定处理)
    base_l = len(os.path.basename(input_path))
    filename = os.path.basename(input_path)[:-4]
    dir = input_path[:-base_l]
    print(f'dir:{dir}\nfilename:{filename}\n')

    proc_tag = f"{dir + filename}__RS_{min_time, v_thres, step}.mp3"
    proc_music = AudioSegment(data=nons_sound.tobytes(), frame_rate=fs, sample_width=int(fw / channels),
                              channels=channels)
    proc_music.export(proc_tag, 'mp3')

    return proc_tag


"""
win_type: 1矩形窗，2汉宁窗，3哈明窗，4布莱克曼窗
"""


def STFT(audio, f_r=44100, win_t=10, overlap_t=5, win_type=1):
    # 最小精度100Hz，最高精度1Hz
    win_l = int(win_t * f_r / 1000) + int(win_t * f_r / 1000) % 2
    overlap_l = int(overlap_t * f_r / 1000)
    cut_length = len(audio) - (len(audio) - win_l) % overlap_l
    # 转换成单声道并截尾，确保二维数组长度对齐
    mono_audio = (audio[:, 0] + audio[:, 1]) / 2
    mono_audio = mono_audio[:cut_length]

    STFT_storage = []

    if win_type == 1:
        for i in range(0, len(mono_audio), overlap_l):
            # 不要更多点数,去除小音量区间也是这个目的
            STFT_storage.append(fft(mono_audio[i:i + win_l], n=win_l))
    elif win_type == 2:
        for i in range(0, len(mono_audio), overlap_l):
            temp = (mono_audio[i:i + win_l]) * 0.5 * (
                        1 + np.cos(2 * np.pi * np.arange(-(win_l // 2), win_l // 2 + 1) / win_l))
            # 不要更多点数,去除小音量区间也是这个目的
            STFT_storage.append(fft(temp, n=win_l))
    elif win_type == 3:
        for i in range(0, len(mono_audio), overlap_l):
            temp = (mono_audio[i:i + win_l]) * (
                        0.54 + 0.46 * np.cos(2 * np.pi * np.arange(-(win_l // 2), win_l // 2 + 1) / win_l))
            # 不要更多点数,去除小音量区间也是这个目的
            STFT_storage.append(fft(temp, n=win_l))
    else:
        for i in range(0, len(mono_audio), overlap_l):
            temp = (mono_audio[i:i + win_l]) * (
                        0.42 + 0.5 * np.cos(2 * np.pi * np.arange(-(win_l // 2), win_l // 2 + 1) / win_l)
                        + 0.08 * np.cos(4 * np.pi * np.arange(-(win_l // 2), win_l // 2 + 1) / win_l))
            # 不要更多点数,去除小音量区间也是这个目的
            STFT_storage.append(fft(temp, n=win_l))
    return np.asarray(STFT_storage, np.complex64)


"""
从input_path加载音频，指定不超过10Hz噪声区间添加标准差为noise_std的高斯或均匀白噪声，作用区域为offset（s）-offset+duration（s）
非静音时段提取窗/FFT单元（绘制语谱图）可以（建议）重叠
添加一定带宽的噪声时，根据带宽决定窗函数长度，选择长度与绘制语谱图的窗长度无关
如果长度过短会出现频带内波纹过大现象，长度过长会出现两端信号强度过小现象
只选择矩形窗函数设计噪声，具有最窄的过渡带带宽，需求在1/10的有效噪声带宽比较合适
如15000-16000Hz噪声，92.2Hz噪声过渡带带宽（44.1kHz对应10ms窗函数）
因此FFT单元win_t不能小于10ms，否则噪声检测精度达不到要求，需要先提高噪声音频长度（采样率固定），是刚需
"""


def addnoise(input_path, noise_avg, noise_std, type, offset, duration, freql=50, freqh=50, step=-1, win_t=10,
             overlap_t=5, win_type=1):
    format = input_path[-3:]
    music = None
    if format == 'wav':
        music = AudioSegment.from_wav(input_path)
    elif format == 'mp3':
        music = AudioSegment.from_mp3(input_path)
    else:
        print("Unsupported audio file type!")
        return

    data = np.asarray(music.get_array_of_samples(), np.float64)
    time = music.duration_seconds
    f_r = music.frame_rate
    f_w = music.frame_width
    channels = music.channels
    # 帧是对于整个音频而言的，不分声道
    # qbits表示量化位数，也就是ffmpeg库里的sample_width
    qbits = len(music.raw_data) / music.frame_count() / channels
    print(music.frame_count() / f_r / time)
    print(f"sampling_rate: {f_r}\nframe_width: {f_w}\nchannels: {channels}\ntime: {time}\nquantized: {qbits}\n")

    data = data.reshape(-1, channels)
    print(data[:100])
    music = data

    # 不直接使用int32和min，max截断措施，考虑到对于施加噪声真实影响的模拟效果
    # float64表示范围更广，不会出现溢出问题

    if offset < 0:
        offset = 0

    # 理想带限噪声的影响包括幅度影响和相位信息，由于真实情况相位信息具有随机性，这里选取offset+i*int(win_t*f_r/1000)作为信号零时刻
    unit = 2 * int(0.46 * f_r * 10 / (freqh - freql) + 1) + 1
    print(f"噪声设计矩形窗长度:{unit}\n")
    if int(duration * f_r) < unit and int(time * f_r) - int(offset * f_r) >= unit:
        print("Error:Duration too short or noise frequency band too thin.")
        duration = unit
    elif int(time * f_r) - int(offset * f_r) < unit:
        print("Error:Offset too large or noise frequency band too thin.")
        duration = unit
        offset = time - unit / f_r
        print(f"offset/duration adjusted to {offset, duration / f_r:.1f}s")
    elif duration > time - offset:
        print(f"Error:Duration/offset too large.Adjusted to {unit / f_r:.1f}s")
        duration = unit
    else:
        duration = int(duration * f_r)
    offset = int(offset * f_r)

    # f_r 最好是 win_l的整数倍，方便画图时使用xticks
    if win_t < 15:
        win_t = 10
    elif win_t < 30:
        win_t = 20
    elif win_t < 45:
        win_t = 40
    elif win_t < 75:
        win_t = 50
    elif win_t < 150:
        win_t = 100
    elif win_t < 225:
        win_t = 200
    elif win_t < 375:
        win_t = 250
    elif win_t < 750:
        win_t = 500
    else:
        win_t = 1000

    noise_part = []
    noise = []
    if freql == 0 and freqh == 'inf':
        if type == 1:
            noise = np.asarray(noise_std * np.random.randn(duration, 2), int)
        else:
            noise = np.random.randint(-np.sqrt(3) * noise_std, high=np.sqrt(3) * noise_std, size=(duration, 2))
        # music[offset:offset + duration] += noise
        noise_part = music[offset:offset + duration] + noise
    else:
        if step == -1:
            # 0.4Hz过渡带、48-52Hz理想带限噪声，unit=2.3s
            step = unit
        # 幅度由概率分布控制，对于每一批帧(step s)作用不同幅值的噪声

        else:
            step = int(step * f_r)

        # 不需要补齐，最后一段如有不足直接按照不足计算
        # if duration%step!=0:
        # duration+=(step-duration%step)
        # print(f"Error:Duration ineligible. Adjusted to {duration/f_r:.1f}")

        noise_part = np.zeros((duration, 2), np.float64)
        noise = np.zeros((duration, 2), np.float64)

        # 对不同带宽的噪声进行强度统一
        unshifted = min(f_r, max(f_r, (freqh - freql))) / f_r * np.sinc((freqh - freql) / f_r *
                                                                        (np.arange(start=-((unit - 1) // 2),
                                                                                   stop=(unit - 1) // 2 + 1)))
        base_noise = []
        # 不需要overlap或者中间补0，对应频域采样不加点数也不减点数
        for i in range(duration):
            base_noise.append([2 * unshifted[i % unit] * np.cos((freql + freqh) / f_r * np.pi * (i % unit)),
                               2 * unshifted[i % unit] * np.cos((freql + freqh) / f_r * np.pi * (i % unit))])

        base_noise = np.asarray(base_noise, np.float32)

        if type == 1:
            for i in range(0, duration, step):
                astep = np.random.randn() * noise_std + noise_avg
                # music[offset+i*step:offset+(i+1)*step]+=astep*base_noise[i*step:(i+1)*step]
                noise_part[i:min(duration, i + step)] = music[offset + i:min(offset + duration,
                                                                             offset + i + step)] + astep * base_noise[
                                                                                                           i:min(
                                                                                                               duration,
                                                                                                               i + step)]
                noise[i:min(duration, i + step)] = astep * base_noise[i:min(duration, i + step)]
        else:
            for i in range(0, duration, step):
                astep = np.random.uniform(-1, 1) * 3 ** 0.5 * noise_std + noise_avg
                # music[offset + i * step:offset + (i + 1) * step] += astep * base_noise[i * step:(i + 1) * step]
                noise_part[i:min(duration, i + step)] = music[offset + i:min(offset + duration,
                                                                             offset + i + step)] + astep * base_noise[
                                                                                                           i:min(
                                                                                                               duration,
                                                                                                               i + step)]
                noise[i:min(duration, i + step)] = astep * base_noise[i:min(duration, i + step)]

    music_clip = music[offset:offset + duration]
    # 转换数据类型为int16（量化位数16）或int32（量化位数24或32），转换为AudioSegment对象，播放并导出音频
    if qbits == 2:
        music_clip = np.asarray(music_clip / np.max(abs(music_clip)) * 2 ** 15 * 0.5, np.int16)
        noise = np.asarray(noise / np.max(abs(noise)) * 2 ** 15 * 0.5, np.int16)
        noise_part = np.asarray(noise_part / np.max(abs(noise_part)) * 2 ** 15 * 0.5, np.int16)
    else:
        music_clip = np.asarray(music_clip / np.max(abs(music_clip)) * 2 ** 31 * 0.5, np.int32)
        noise = np.asarray(noise / np.max(abs(noise)) * 2 ** 31 * 0.5, np.int32)
        noise_part = np.asarray(noise_part / np.max(abs(noise_part)) * 2 ** 31 * 0.5, np.int32)

    # 音频文件导出（同输入文件目录）
    base_l = len(os.path.basename(input_path))
    filename = os.path.basename(input_path)[:-4]
    dir = input_path[:-base_l]
    print(f'dir:{dir}\nfilename:{filename}\n')

    noise_music = AudioSegment(noise_part.tobytes(), frame_rate=f_r, sample_width=int(f_w / channels),
                               channels=channels)
    nm_tag = dir + f"noisemusic({round(offset / f_r)}-{round((offset + duration) / f_r)}_{type}_f{freql}_f{freqh}_{noise_avg}_{noise_std}_{int(step / f_r + 0.5)}).mp3"
    n_tag = dir + f"noise({type}_f{freql}_f{freqh}_{noise_avg}_{noise_std}_{int(step / f_r + 0.5)}).mp3"
    m_tag = dir + f'music_seg({round(offset / f_r)}-{round((offset + duration) / f_r)}).mp3'
    noise_music.export(nm_tag, format='mp3')
    pnoise = AudioSegment(noise.tobytes(), frame_rate=f_r, sample_width=int(f_w / channels), channels=channels)
    pnoise.export(n_tag, format='mp3')
    # music_clip=music[offset:offset+duration]
    music_seg = AudioSegment(music_clip.tobytes(), frame_rate=f_r, sample_width=int(f_w / channels), channels=channels)
    music_seg.export(m_tag, 'mp3')

    # play(noise_music)
    # play(pnoise)

    print(music[offset:offset + duration].shape)

    # 频谱分析
    M = STFT(music[offset:offset + duration], f_r, win_t, overlap_t, win_type)
    MN = STFT(noise_part, f_r, win_t, overlap_t, win_type)
    N = STFT(noise, f_r, win_t, overlap_t, win_type)

    # 由于输入音频信号是实信号，频谱具有对称性，因此截取一半绘图
    # 下面的截取方法奇数偶数点FFT都适用
    M = abs(M[:, :int(len(M[0]) / 2) + 1])
    MN = abs(MN[:, :int(len(MN[0]) / 2) + 1])
    N = abs(N[:, :int(len(N[0]) / 2) + 1])

    # 规范化成uint8类型
    M = np.asarray(np.minimum(255 * np.ones_like(M), M / np.mean(M) * 128), np.uint8).T
    MN = np.asarray(np.minimum(255 * np.ones_like(MN), MN / np.mean(MN) * 128), np.uint8).T
    N = np.asarray(N / np.max(N) * 255, np.uint8).T
    # M = np.asarray(M/np.max(M)*255, np.uint8).T
    # MN = np.asarray(MN/np.max(MN)*255, np.uint8).T

    # 计算差异
    # 绘制伪彩色语谱图
    img_ss(M, MN, N, freql, freqh, f_r, win_t, overlap_t, offset, duration, type, noise_avg, noise_std, step)

    # 返回输出相对路径
    return m_tag, n_tag, nm_tag, M


"""
输入处理前/处理后规范化到uint8的STFT二维幅度数组、带阻区间参数
"""


def img_ss(M, MN, N, freql, freqh, f_r, win_t, overlap_t, offset=None, duration=None, type=None, noise_avg=None,
           noise_std=None, step=None):
    # 计算差异
    # FFT精度：1000ms/win_t(ms) Hz
    error = (np.sum(np.square(MN - M)) / len(M[0]) / ((freqh - freql + 1e-7) / (1000 / win_t))) ** 0.5
    print(f"error: {error}")
    # 语谱图pyplot绘制（DIP伪彩色处理）+ 图像保存
    MMN = pseudocolor_enhance([M, MN, N], 1, 5, 1)
    low_ind = min(int(max(0, (freql - 50)) * win_t / 1000), int(max(0, (freql - 2 * (freqh - freql))) * win_t / 1000))
    high_ind = max(int(min(f_r / 2, (freqh + 50)) * win_t / 1000),
                   int(min(f_r / 2, (freqh + 2 * (freqh - freql))) * win_t / 1000))
    if noise_avg != None:
        cv2.imwrite(f"orig_({round(offset / f_r)}-{round((offset + duration) / f_r)}).jpg", MMN[0][low_ind:high_ind, :])
        cv2.imwrite(
            f"noisemusic_({round(offset / f_r)}-{round((offset + duration) / f_r)} {type} f{freql} f{freqh} {noise_avg} {noise_std} {int(step / f_r + 0.5)}).jpg",
            MMN[1][low_ind:high_ind, :])
        cv2.imwrite(f"noise_({type} f{freql} f{freqh} {noise_avg} {noise_std} {int(step / f_r + 0.5)}).jpg",
                    MMN[2][low_ind:high_ind, :])
        print("Writing Success1")
    else:
        # cv2.imwrite(f"noisemusic_(0-{round(len(M[0])*win_t/1000)}).jpg",MMN[0][low_ind:high_ind,:])
        cv2.imwrite(f"proc_(0-{round(len(M[0]) * win_t / 1000)} f{freql} f{freqh}).jpg", MMN[1][low_ind:high_ind, :])
        print("Writing Success2")

    M2 = cv2.cvtColor(MMN[0], cv2.COLOR_BGR2RGB)
    MN2 = cv2.cvtColor(MMN[1], cv2.COLOR_BGR2RGB)
    N2 = cv2.cvtColor(MMN[2], cv2.COLOR_BGR2RGB)

    # 语谱图灰度图绘制
    plt.figure()
    plt.subplot(211)
    plt.imshow(M2, extent=[0, len(M2[0]), len(M2), 0], aspect='auto')
    if noise_avg != None:
        # plt.xticks(range(0,len(M2[0]),1000//overlap_t),[f'{i/f_r}' for i in range(offset,offset+duration,f_r)])
        if duration % f_r != 0:
            plt.xticks(range(0, (1 + duration // f_r) * (1000 // overlap_t), 1000 // overlap_t),
                       [f'{i / f_r}' for i in range(offset, offset + duration, f_r)])
        else:
            plt.xticks(range(0, duration // f_r * (1000 // overlap_t), 1000 // overlap_t),
                       [f'{i / f_r}' for i in range(offset, offset + duration, f_r)])
    elif len(M2[0]) % (1000 // overlap_t) != 0:
        plt.xticks(range(0, len(M2[0]), 1000 // overlap_t),
                   [f'{i}' for i in range(0, int(len(M2[0]) / (1000 // overlap_t)) + 1, 1)])
    else:
        plt.xticks(range(0, len(M2[0]), 1000 // overlap_t),
                   [f'{i}' for i in range(0, int(len(M2[0]) / (1000 // overlap_t)), 1)])

    plt.yticks(range(0, len(M2) - len(M2) % 4, len(M2) // 4), [f'{j / 4 * f_r / 2}' for j in range(0, 4)])
    # plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.title(f'STFT-Original Audio | win_t: {win_t}ms')
    plt.subplot(212)
    plt.imshow(MN2, extent=[0, len(MN2[0]), len(MN2), 0], aspect='auto')
    if noise_avg != None:
        # plt.xticks(range(0,len(M2[0]),1000//overlap_t),[f'{i/f_r}' for i in range(offset,offset+duration,f_r)])
        if duration % f_r != 0:
            plt.xticks(range(0, (1 + duration // f_r) * (1000 // overlap_t), 1000 // overlap_t),
                       [f'{i / f_r}' for i in range(offset, offset + duration, f_r)])
        else:
            plt.xticks(range(0, duration // f_r * (1000 // overlap_t), 1000 // overlap_t),
                       [f'{i / f_r}' for i in range(offset, offset + duration, f_r)])

    elif len(M2[0]) % (1000 // overlap_t) != 0:
        plt.xticks(range(0, len(M2[0]), 1000 // overlap_t),
                   [f'{i}' for i in range(0, int(len(M2[0]) / (1000 // overlap_t)) + 1, 1)])
    else:
        plt.xticks(range(0, len(M2[0]), 1000 // overlap_t),
                   [f'{i}' for i in range(0, int(len(M2[0]) / (1000 // overlap_t)), 1)])

    plt.yticks(range(0, len(M2) - len(M2) % 4, len(M2) // 4), [f'{j / 4 * f_r / 2}' for j in range(0, 4)])
    plt.xlabel('time (s)')
    plt.ylabel('frequency (Hz)')
    plt.title(f'STFT-Processed Audio | win_t: {win_t}ms')

    if noise_avg != None:
        plt.figure()
        plt.imshow(N2, extent=[0, len(N2[0]), len(N2), 0], aspect='auto')
        # plt.xticks(range(0, len(M2[0]), 1000 // overlap_t), [f'{i / f_r}' for i in range(offset, offset + duration, f_r)])
        if duration % f_r != 0:
            plt.xticks(range(0, (1 + duration // f_r) * (1000 // overlap_t), 1000 // overlap_t),
                       [f'{i / f_r}' for i in range(offset, offset + duration, f_r)])
        else:
            plt.xticks(range(0, duration // f_r * (1000 // overlap_t), 1000 // overlap_t),
                       [f'{i / f_r}' for i in range(offset, offset + duration, f_r)])
        plt.yticks(range(0, len(M2) - len(M2) % 4, len(M2) // 4), [f'{j / 4 * f_r / 2}' for j in range(0, 4)])
        plt.xlabel('time (s)')
        plt.ylabel('frequency (Hz)')
        plt.title(f'STFT-Noise Audio | win_t: {win_t}ms')

    plt.show()


"""
去噪流程：非静音时段提取->非理想窄带带阻滤波->重映射
仅使用FIR-I型 Hanning窗法设计带阻滤波器
trans_bw measured in Hz
win_t stands for time length of non-overlapping convolution unit(ms)
过渡带宽在1/10通带带宽比较合适，如果精度受限可以先提高噪声音频长度（采样率固定），不是刚需
如15000-16000Hz噪声，100Hz过渡带带宽
"""


def filter(input_path, input_path2, freql, freqh, win_t1=20, overlap_t=10, win_t2=200, win_type=1, remove=False,
           slice=False):
    # 加载音频
    # input_path1 无噪，input_path2 有噪
    format1 = input_path[-3:]
    format2 = input_path2[-3:]
    orig_music, noise_music = None, None
    if format1 == 'wav':
        orig_music = AudioSegment.from_wav(input_path)
    elif format1 == 'mp3':
        orig_music = AudioSegment.from_mp3(input_path)
    else:
        print("File1: Unsupported audio file type!")
        return

    if format2 == 'wav':
        noise_music = AudioSegment.from_wav(input_path2)
    elif format2 == 'mp3':
        noise_music = AudioSegment.from_mp3(input_path2)
    else:
        print("File2: Unsupported audio file type!")
        return

    channels = noise_music.channels
    fs, fw = noise_music.frame_rate, noise_music.frame_width
    data = np.asarray(noise_music.get_array_of_samples(), np.float64)
    data = data.reshape(-1, channels)

    trans_bw = min(min(freql, fs / 2 - freqh), max((freqh - freql) / 10, 3.11 * np.pi / (
                2 * np.pi * int(((fs / (1000 / win_t1)) + 1) // 2 - 1)) * fs))
    if trans_bw == min(freql, fs / 2 - freqh):
        if 1.555 / trans_bw * fs - int(1.555 / trans_bw * fs) > 0:
            win_t1 = int(1000 / (fs / (2 * int(1.555 / trans_bw * fs + 1) + 1)))

    print(f"win_t1: {win_t1}ms")

    win_l1 = int(win_t1 / 1000 * fs)
    if len(data) % win_l1 != 0:
        data = data[:-(len(data) % win_l1)]

    data2 = np.zeros_like(data)

    if remove:
        # min_time 设置为 win_t2，step 设置为win_t2/100,
        # 这样可以（1）降低音频信号小音量分割“受伪”损失
        # （2）避免拼接产生的“爆音”现象
        data2, inds = remove_small_volumes(data, win_t2, 40, win_t2 / 100, fs)
        # 滤波处理-计算分离静音区间后的作用长度
        # 在【STFT点数，过渡带参数控制】之间取最小值，信号长度不超过输入信号x长度
        # 精度与不同时间帧频率成分变化之间的矛盾（卷积的长度限制规则是不可动摇的）
        M = min(int(((fs / (1000 / win_t1)) + 1) // 2 - 1), int(3.11 * np.pi / (trans_bw / fs * 2 * np.pi)))
        if M != int(((fs / (1000 / win_t1)) + 1) // 2 - 1) and 3.11 * np.pi / (trans_bw / fs * 2 * np.pi) - int(
                3.11 * np.pi / (trans_bw / fs * 2 * np.pi)) > 0:
            M += 1
        print(f"带阻滤波器Hanning窗点数:{2 * M + 1}\n满足要求:{not M == int(((fs / (1000 / win_t1)) + 1) // 2 - 1)}")
        print(f"过渡带带宽:{trans_bw}\n")
        # arange不包括stop尾部
        hanning_window = 0.5 * (1 + np.cos(2 * np.pi * np.arange(-M, M + 1) / (2 * M + 1)))
        lpw = (freql - trans_bw / 2) / fs * 2 * np.pi
        hpw = (freqh + trans_bw / 2) / fs * 2 * np.pi
        lp, hp = np.zeros((2 * M + 1)), np.zeros((2 * M + 1))
        lp[:M] = np.sin(lpw * np.arange(-M, 0)) / np.pi / np.arange(-M, 0)
        lp[M] = (freql - trans_bw / 2) / fs * 2
        lp[M + 1:] = np.sin(lpw * np.arange(1, M + 1)) / np.pi / np.arange(1, M + 1)
        hp[:M] = -np.sin(hpw * np.arange(-M, 0)) / np.pi / np.arange(-M, 0)
        hp[M + 1:] = -np.sin(hpw * np.arange(1, M + 1)) / np.pi / np.arange(1, M + 1)
        hp[M] = 1 - (freqh + trans_bw / 2) / fs * 2

        bs = lp + hp
        bs = np.asarray([bs * hanning_window, bs * hanning_window]).T

        plt.figure()
        plt.plot([i / (2 * M + 1) * fs for i in range(2 * M + 1)], 20 * np.log10(abs(fft(bs[:, 0]))), 'b-')
        # plt.show()

        bs_tag = f"bandstop_(order{2 * M} f{freql} f{freqh}).wav"
        bsi = np.asarray(bs / np.max(abs(bs)) * 2 ** 15 * 0.5, np.uint16)
        if fw / channels == 3 or fw / channels == 4:
            bsi = np.asarray(bs / np.max(abs(bs)) * 2 ** 31 * 0.5, np.uint32)
        bs_audio = AudioSegment(data=bsi.tobytes(), frame_rate=fs, sample_width=int(fw / channels), channels=channels)
        bs_audio.export(bs_tag, 'wav')

        # 使用手编卷积函数
        # data2 = convolution(data2, bs)
        # 使用numpy库卷积函数
        if slice:
            for i in range(0, len(data2), win_l1):
                data2[i:i + win_l1, 0] = np.convolve(data2[i:i + win_l1, 0], bs[:, 0], 'full')[M:-M]
                data2[i:i + win_l1, 1] = np.convolve(data2[i:i + win_l1, 1], bs[:, 0], 'full')[M:-M]
        else:
            data2[:, 0] = np.convolve(data2[:, 0], bs[:, 0], 'full')[M:-M]
            data2[:, 1] = np.convolve(data2[:, 1], bs[:, 0], 'full')[M:-M]

        # 重映射
        # 动态更新的循环条件使用while
        i = 0
        while i < len(data2):
            if len(inds) > 0 and i == inds[0][0] - 1:
                data2 = np.concatenate((data2[:i + 1], np.zeros((inds[0][1] - inds[0][0] + 1, 2)), data2[i + 1:]),
                                       axis=0)
                del inds[0]
            else:
                i += 1
        print(data.shape)
        print(data2.shape)
        assert (data2.shape == data.shape)

    else:
        M = min(int(((fs / (1000 / win_t1)) + 1) // 2 - 1), int(3.11 * np.pi / (trans_bw / fs * 2 * np.pi)))
        if M != int(((fs / (1000 / win_t1)) + 1) // 2 - 1) and 3.11 * np.pi / (trans_bw / fs * 2 * np.pi) - int(
                3.11 * np.pi / (trans_bw / fs * 2 * np.pi)) > 0:
            M += 1
        print(f"带阻滤波器Hanning窗点数:{2 * M + 1}\n满足要求:{not M == int(((fs / (1000 / win_t1)) + 1) // 2 - 1)}")
        print(f"过渡带带宽:{trans_bw}\n")
        # arange不包括stop尾部
        hanning_window = 0.5 * (1 + np.cos(2 * np.pi * np.arange(-M, M + 1) / (2 * M + 1)))
        lpw = max(0, (freql - trans_bw / 2)) / fs * 2 * np.pi
        hpw = min(fs / 2, (freqh + trans_bw / 2)) / fs * 2 * np.pi
        lp, hp = np.zeros((2 * M + 1)), np.zeros((2 * M + 1))
        lp[:M] = np.sin(lpw * np.arange(-M, 0)) / np.pi / np.arange(-M, 0)
        lp[M] = lpw / np.pi
        lp[M + 1:] = np.sin(lpw * np.arange(1, M + 1)) / np.pi / np.arange(1, M + 1)
        hp[:M] = -np.sin(hpw * np.arange(-M, 0)) / np.pi / np.arange(-M, 0)
        hp[M + 1:] = -np.sin(hpw * np.arange(1, M + 1)) / np.pi / np.arange(1, M + 1)
        hp[M] = 1 - hpw / np.pi

        bs = lp + hp
        bs = np.asarray([bs * hanning_window, bs * hanning_window]).T

        bs_tag = f"bandstop_(order{2 * M} f{freql} f{freqh}).wav"
        bsi = np.asarray(bs / np.max(abs(bs)) * 2 ** 15 * 0.5, np.uint16)
        if fw / channels == 3 or fw / channels == 4:
            bsi = np.asarray(bs / np.max(abs(bs)) * 2 ** 31 * 0.5, np.uint32)
        bs_audio = AudioSegment(data=bsi.tobytes(), frame_rate=fs, sample_width=int(fw / channels), channels=channels)
        bs_audio.export(bs_tag, 'wav')

        # 使用手编卷积函数
        # data2 = convolution(data, bs)
        # 使用numpy库卷积函数
        if slice:
            for i in range(0, len(data2), win_l1):
                data2[i:i + win_l1, 0] = np.convolve(data[i:i + win_l1, 0], bs[:, 0], 'full')[M:-M]
                data2[i:i + win_l1, 1] = np.convolve(data[i:i + win_l1, 1], bs[:, 0], 'full')[M:-M]

        else:
            data2[:, 0] = np.convolve(data[:, 0], bs[:, 0], 'full')[M:-M]
            data2[:, 1] = np.convolve(data[:, 1], bs[:, 0], 'full')[M:-M]

    # 去除直流分量
    data2 = data2 - np.mean(data2)

    # 转换数据类型为int16（量化位数16）或int32（量化位数24或32），转换为AudioSegment对象，播放并导出音频
    if fw / channels == 2:
        data2 = np.asarray(data2 / np.max(abs(data2)) * 2 ** 15 * 0.7, np.int16)
    else:
        data2 = np.asarray(data2 / np.max(abs(data2)) * 2 ** 31 * 0.7, np.int32)

    # 支持同目录导出
    base_l = len(os.path.basename(input_path))
    filename = os.path.basename(input_path)[:-4]
    dir = input_path[:-base_l]
    print(f'dir:{dir}\nfilename:{filename}\n')

    proc_tag = f"{dir + filename}(0-{round(len(data) / fs)}_f{freql}_f{freqh}_r{remove})__FIRf.mp3"
    proc_music = AudioSegment(data=data2.tobytes(), frame_rate=fs, sample_width=int(fw / channels), channels=channels)
    proc_music.export(proc_tag, 'mp3')

    # play(proc_music)

    # 先转置
    MN = abs(STFT(data, fs, win_t1, overlap_t, win_type))
    M2 = abs(STFT(data2, fs, win_t1, overlap_t, win_type))
    assert (M2.shape == MN.shape)

    M2 = M2[:, 0:int(len(M2[0]) / 2) + 1].T
    MN = MN[:, 0:int(len(MN[0]) / 2) + 1].T

    MN = np.asarray(np.minimum(255 * np.ones_like(MN), MN / np.mean(MN) * 128), np.uint8)
    M2 = np.asarray(np.minimum(255 * np.ones_like(M2), M2 / np.mean(M2) * 128), np.uint8)

    img_ss(MN, M2, np.ones_like(MN), freql, freqh, fs, win_t1, overlap_t)

    # 和加噪前比较
    data3 = np.asarray(orig_music.get_array_of_samples()).reshape(-1, channels)

    # M 时间长度一般更长（因为M对应滤波后信号，一般会有结尾处理）
    M = abs(STFT(data3, fs, win_t1, overlap_t, win_type))
    M = M[:, :int(len(M[0]) / 2) + 1].T[:, :len(M2[0])]
    M = np.asarray(np.minimum(M / np.mean(M) * 128, 255 * np.ones_like(M)), np.uint8)
    img_ss(M, M2, np.ones_like(M2), freql, freqh, fs, win_t1, overlap_t)

    return proc_tag, M


"""
注意是对二通道数组计算圆周卷积
"""


def convolution(x, h):
    if len(h) % 2 == 0:
        print("Invalid length of signal h")
        return
    if len(h) > len(x):
        print("Short audio segment ignored.")
        return

    M = (len(h) - 1) // 2
    # print(M)
    y = np.zeros_like(x)
    # [0,M-1] [M,win_l-M-1] [win_l-M,win_l-1]
    # 限制M的大小小于x的长度的一半，减少情况的讨论数
    for i in range(M):
        y[i] = np.sum(h[M + i::-1] * x[:M + i + 1], axis=0)
    for i in range(M, len(x) - M):
        y[i] = np.sum(h[::-1] * x[i - M:i + M + 1], axis=0)
    for i in range(len(x) - M, len(x)):
        y[i] = np.sum(h[:(i - len(x) + M):-1] * x[i - M:], axis=0)

    return y


def echoing(input_path, a1, a2, a3, t1, t2, t3, win_t=10, overlap_t=5, win_type=1, TD=True, speed=True, region_t=400,
            low_order=2):
    # 加载音频
    format = input_path[-3:]
    music = None
    if format == 'wav':
        music = AudioSegment.from_wav(input_path)
    elif format == 'mp3':
        music = AudioSegment.from_mp3(input_path)
    else:
        filename = os.path.basename(input_path)
        real_format = filename[max(-8, len(filename)):].strip('.')[1]
        print(f"File: Unsupported audio file type {real_format}!")
        return

    channels = music.channels
    fs, fw = music.frame_rate, music.frame_width
    sound = np.asarray(music.get_array_of_samples(), np.float64)
    sound = sound.reshape(-1, channels)

    region_l = int(region_t / 1000 * fs)
    # 延时单元-延时点数转换
    R1 = int(t1 / 1000 * fs)
    R2 = int(t2 / 1000 * fs)
    R3 = int(t3 / 1000 * fs)

    # 两（三）种滤波器都是实信号因果滤波器(而且h[0]≠0),笔者编写的圆周卷积函数只适用于FIR-I/III型零相位滤波器
    # 限制滤波器时域序列长度不能超过数据序列长度的一半(一次回波滑动在窗的一半长度之内，确保至少有两次回波）
    if speed:
        if R1 + 1 > region_l // int(low_order):
            R1 = region_l // int(low_order) - 1
        if R2 > region_l // int(low_order):
            R2 = region_l // int(low_order)
        if R3 > region_l // int(low_order):
            R3 = region_l // int(low_order)
    else:
        if R1 + 1 > len(sound) // int(low_order):
            R1 = len(sound) // int(low_order) - 1
        if R2 > len(sound) // int(low_order):
            R2 = len(sound) // int(low_order)
        if R3 > len(sound) // int(low_order):
            R3 = len(sound) // int(low_order)

    assert (a1 > 0 and a1 < 1)
    assert (a2 > 0 and a2 < 1)
    assert (a3 > 0 and a3 < 1)

    sound_comb, sound_2comb, sound_allp = np.zeros_like(sound), np.zeros_like(sound), np.zeros_like(sound)
    if TD:
        # FIR 精简实现（不使用卷积）
        sound_comb[:R1] = sound[:R1]
        for i in range(R1, len(sound)):
            sound_comb[i] = sound[i] - a1 * sound[i - R1]

        # Comb Filter （IIR滤波器）- 时域实现
        sound_2comb[:R2] = sound[:R2]
        for i in range(R2, len(sound)):
            sound_2comb[i] = a2 * sound_2comb[i - R2] + sound[i]

        # All-pass Filter （IIR滤波器）- 时域实现
        sound_allp[:R3] = -a3 * sound[:R3]
        for i in range(R3, len(sound)):
            sound_allp[i] = a3 * sound_allp[i - R3] - a3 * sound[i] + sound[i - R3]

    else:
        # 前馈型梳妆滤波器-FIR，直接时域设计
        h1 = []
        if R1 == 1:
            h1 = np.asarray([1, a1])
        else:
            h1 = np.concatenate((np.ones(1), np.zeros((R1 - 1)), a1 * np.ones(1)))

        print(f"R: {R1}")

        fft_n = 5 * max(max(R1, R2), R3)
        H1 = fft(h1, n=fft_n)

        # 反馈型梳妆滤波器-IIR，频域设计(FIR近似)
        H2 = 1 / (1 - a2 * np.exp(-1j * 2 * np.pi * np.arange(0, fft_n) / fft_n * R2))
        h2 = ifft(H2)

        # 全通滤波器-IIR，频域设计(FIR近似)
        H3 = ((-a3 + np.exp(-1j * 2 * np.pi * np.arange(0, fft_n) / fft_n * R3))
              / (1 - a3 * np.exp(-1j * 2 * np.pi * np.arange(0, fft_n) / fft_n * R3)))
        h3 = ifft(H3)

        plt.figure()
        plt.subplot(311)
        plt.plot([i * fs / fft_n for i in range(0, fft_n)], abs(H1), 'b-')
        plt.subplot(312)
        plt.plot([i * fs / fft_n for i in range(0, fft_n)], abs(H2), 'g--')
        plt.subplot(313)
        plt.plot([i * fs / fft_n for i in range(0, fft_n)], np.angle(H3, deg=False), 'r-.')

        plt.show()

        if speed:
            for i in range(0, len(sound), region_l):
                sound_comb[i:min(i + region_l, len(sound)), 0] = np.convolve(sound[i:min(i + region_l, len(sound)), 0],
                                                                             h1, 'full')[:min(region_l, len(sound) - i)]
                sound_comb[i:min(i + region_l, len(sound)), 1] = np.convolve(sound[i:min(i + region_l, len(sound)), 1],
                                                                             h1, 'full')[:min(region_l, len(sound) - i)]
        else:
            sound_comb[:, 0] = np.convolve(sound[:, 0], h1, 'full')
            sound_comb[:, 1] = np.convolve(sound[:, 1], h1, 'full')

        if speed:
            for i in range(0, len(sound), region_l):
                sound_2comb[i:min(i + region_l, len(sound)), 0] = np.convolve(sound[i:min(i + region_l, len(sound)), 0],
                                                                              h2, 'full')[
                                                                  :min(region_l, len(sound) - i)]
                sound_2comb[i:min(i + region_l, len(sound)), 1] = np.convolve(sound[i:min(i + region_l, len(sound)), 1],
                                                                              h2, 'full')[
                                                                  :min(region_l, len(sound) - i)]
        else:
            sound_2comb[:, 0] = np.convolve(sound[:, 0], h2, 'full')
            sound_2comb[:, 1] = np.convolve(sound[:, 1], h2, 'full')

        if speed:
            for i in range(0, len(sound), region_l):
                sound_allp[i:min(i + region_l, len(sound)), 0] = np.convolve(sound[i:min(i + region_l, len(sound)), 0],
                                                                             h3, 'full')[:min(region_l, len(sound) - i)]
                sound_allp[i:min(i + region_l, len(sound)), 1] = np.convolve(sound[i:min(i + region_l, len(sound)), 1],
                                                                             h3, 'full')[:min(region_l, len(sound) - i)]
        else:
            sound_allp[:, 0] = np.convolve(sound[:, 0], h3, 'full')
            sound_allp[:, 1] = np.convolve(sound[:, 1], h3, 'full')

        if fw / channels == 2:
            sound_comb = np.asarray(sound_comb / np.max(abs(sound_comb)) * 2 ** 15 * 0.5, np.int16)
            sound_2comb = np.asarray(sound_2comb / np.max(abs(sound_2comb)) * 2 ** 15 * 0.5, np.int16)
            sound_allp = np.asarray(sound_allp / np.max(abs(sound_allp)) * 2 ** 15 * 0.5, np.int16)
        elif fw / channels == 3 or fw / channels == 4:
            sound_comb = np.asarray(sound_comb / np.max(abs(sound_comb)) * 2 ** 31 * 0.5, np.int32)
            sound_2comb = np.asarray(sound_2comb / np.max(abs(sound_2comb)) * 2 ** 31 * 0.5, np.int32)
            sound_allp = np.asarray(sound_allp / np.max(abs(sound_allp)) * 2 ** 31 * 0.5, np.int32)

    # 支持同目录输出新文件
    file_name = os.path.basename(input_path)[:-4]
    tail_l = len(file_name) + 4
    dir_path = input_path[:-tail_l]
    print(f"dir_path:{dir_path}")

    comb = AudioSegment(data=sound_comb.tobytes(), frame_rate=fs, sample_width=int(fw / channels), channels=channels)
    sharp_comb = AudioSegment(data=sound_2comb.tobytes(), frame_rate=fs, sample_width=int(fw / channels),
                              channels=channels)
    allp = AudioSegment(data=sound_allp.tobytes(), frame_rate=fs, sample_width=int(fw / channels), channels=channels)

    comb_p = dir_path + file_name + '_comb.mp3'
    scomb_p = dir_path + file_name + '_scomb.mp3'
    allp_p = dir_path + file_name + '_allp.mp3'
    comb.export(comb_p, 'mp3')
    sharp_comb.export(scomb_p, 'mp3')
    allp.export(allp_p, 'mp3')

    # 观察STFT语谱图
    M = abs(STFT(sound, fs, win_t, overlap_t, win_type))
    C = abs(STFT(sound_comb, fs, win_t, overlap_t, win_type))
    C2 = abs(STFT(sound_2comb, fs, win_t, overlap_t, win_type))
    A = abs(STFT(sound_allp, fs, win_t, overlap_t, win_type))

    # 由于输入音频信号是实信号，频谱具有对称性，因此截取一半绘图
    # 下面的截取方法奇数偶数点FFT都适用
    M = M[:, :len(M[0]) // 2 + 1].T
    C = C[:, :len(C[0]) // 2 + 1].T
    C2 = C2[:, :len(C2[0]) // 2 + 1].T
    A = A[:, :len(A[0]) // 2 + 1].T

    # 规范化成uint8类型
    M = np.asarray(np.minimum(255 * np.ones_like(M), M / np.mean(M) * 128), np.uint8)
    C = np.asarray(np.minimum(255 * np.ones_like(C), C / np.mean(C) * 128), np.uint8)
    C2 = np.asarray(np.minimum(255 * np.ones_like(C2), C2 / np.mean(C2) * 128), np.uint8)
    A = np.asarray(np.minimum(255 * np.ones_like(A), A / np.mean(A) * 128), np.uint8)

    img_ss(M, C, np.ones_like(M), 0, fs / 2, fs, win_t, overlap_t)
    img_ss(M, C2, np.ones_like(M), 0, fs / 2, fs, win_t, overlap_t)
    img_ss(M, A, np.ones_like(M), 0, fs / 2, fs, win_t, overlap_t)

    return comb_p, scomb_p, allp_p, M, C, C2, A


"""
window_t serves to slice audio into short time pieces and cut down calculation time, measured in miliseconds
"""


def equalizer(input_path, DBGains, Q=2 ** (0.5), N=512, speed=True, window_t=400):
    # As计算
    As = np.asarray(np.power(10, np.asarray(DBGains, int) / 40), float)
    # 加载音频
    format = input_path[-3:]
    music = None
    if format == 'wav':
        music = AudioSegment.from_wav(input_path)
    elif format == 'mp3':
        music = AudioSegment.from_mp3(input_path)
    else:
        print("Unsupported audio file type!")
        return

    fs, fw, channels = music.frame_rate, music.frame_width, music.channels
    # 精度提升（后续卷积计算需要）+声道调整
    data = np.asarray(music.get_array_of_samples(), np.float32).reshape(-1, channels)
    freq_Rs = []
    freq_highest = 16000
    for i in range(10):
        # 网上的“工程师公式”结论似乎有一些问题
        """
        w0=2*np.pi*freq_highest/pow(2,i)/fs
        alpha=np.sin(w0)/2/Q
        a0=1+alpha/As[-i-1]
        a1=-2*np.cos(w0)
        a2=1-alpha/As[-i-1]
        b0=1+alpha*As[-i-1]
        b1=-2*np.cos(w0)
        b2=1-alpha*As[-i-1]

        # whole=False对应一半采样率的上界（实系统滤波器频谱具有对称性）
        # 但是ifft需要完整的频谱采样，所以选择whole=True
        _,h=signal.freqz(b=[b0,b1,b2],a=[a0,a1,a2],worN=N,whole=True,fs=fs)

        H=fft(h)
        """
        # 频率响应出发
        fc = freq_highest / pow(2, i)
        H = (-(np.arange(0, N) / N * fs) ** 2 + 1j * As[-i - 1] / Q * np.arange(0, N) / N * fs * fc + fc ** 2) / (
                -(np.arange(0, N) / N * fs) ** 2 + 1j / As[-i - 1] / Q * np.arange(0, N) / N * fs * fc + fc ** 2)

        freq_Rs.append(H)

    # 频率响应（幅度谱）可视化，彩色绘图
    _, figs = plt.subplots(2, 5)
    import matplotlib.colors as mcolors
    colors = tuple(k for k in mcolors.CSS4_COLORS.keys())
    # colors=colors[0:len(colors):(len(colors)//9 if len(colors)%9>0 else len(colors)//9-1)]
    colors = np.random.choice(colors, size=(10), replace=True)
    print(f"colors:{colors}\n")

    for i in range(0, 10, 1):
        figs[i // 5][i % 5].plot([j / N * fs for j in range(N // 2 + 1)],
                                 20 * np.log10(abs(freq_Rs[9 - i][:(N // 2 + 1)])), c=colors[i])
        figs[i // 5][i % 5].set_title(f'f0:{int(freq_highest / pow(2, 9 - i))}Hz\nA:{As[i]:.2f} Q:{Q:.2f} N:{N}')

    # final_eqt=np.sum([h for h in filters],axis=0)
    # final_eqFR=np.sum([H for H in freq_Rs],axis=0)
    final_eqFR = np.ones((N,), np.complex64)
    for H in freq_Rs:
        final_eqFR *= H

    final_eqt = ifft(final_eqFR, n=N)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot([i / N * fs for i in range(N // 2 + 1)], 20 * np.log10(abs(final_eqFR[:(N // 2 + 1)])), 'm-')
    plt.title(f'Q:{Q:.2f} N:{N}')
    plt.subplot(2, 1, 2)
    plt.stem(range(N), np.real(final_eqt), 'r-o')
    plt.title(f'Q:{Q:.2f} N:{N}')

    plt.show()

    # 逐频段、逐时段、逐声道（统一）处理
    # 注意h可能是复数信号，但是y一定是实数信号，所以要做取实部操作
    eq_data = np.zeros_like(data)
    if speed:
        win_l = int(window_t / 1000 * fs)
        for l in range(0, len(data), win_l):
            for ch in range(channels):
                eq_data[l:min(l + win_l, len(data)), ch] = np.real(np.convolve(data[l:min(l + win_l, len(data)), ch],
                                                                               np.real(final_eqt), 'full')[
                                                                   :min(win_l, len(data) - l)])
    else:
        for ch in range(channels):
            eq_data[:, ch] = np.real(np.convolve(data[:, ch], np.real(final_eqt), 'full')[:len(data)])

    # 音频处理与导出

    # 16-bit quantization
    if fw / channels == 2:
        # 最大音量调整到支持最大音量的70%
        eq_data = np.asarray(eq_data / np.max(abs(eq_data)) * np.max(abs(data)), np.int16)
    else:
        # 尽管24比特量化精度未达到32，但是放缩处理只改变了步长<->绝对精度，不会改变相对精度，相当于音量拉低拉高
        # 导出的时候还是分开导出
        eq_data = np.asarray(eq_data / np.max(abs(eq_data)) * np.max(abs(data)), np.int32)

    assert (np.max(abs(eq_data)) == np.max(abs(data)))

    # 支持同目录导出
    base_l = len(os.path.basename(input_path))
    filename = os.path.basename(input_path)[:-4]
    dir = input_path[:-base_l]
    output_path = dir + filename + f'__EQ_A{np.asarray(DBGains, np.int16)}_Q{Q:.2f}_N{N}.mp3'
    print(f'dir:{dir}\nfilename:{filename}\noutput_filepath{output_path}\n')
    export_audio = AudioSegment(data=eq_data.tobytes(), frame_rate=fs, sample_width=int(fw / channels),
                                channels=channels)
    export_audio.export(output_path, 'mp3')

    return output_path


# I型 Comb-Filter级联:
# 高通滤波器:0<a<1,R=1;低通滤波器:-1<a<0,R=1

# 或者使用IIR设计，FIR-I型设计
# 也可以通过峰值/陷波滤波器(peak/notch filter)实现更精细的频移调整
# def freq_shifter(input_path):


# 自适应FIR-N阶滤波器，梯度下降优化
# IIR滤波器会出现数值爆炸的情况，即不满足BIBO，最外极点在单位圆外（或圆上）
def filter_self_adaptive(input_path1, input_path2, N=19, batch_size=32, alpha_initial=0.1, thres=0.01,
                         method='sgd', rho=0.9, filter=None, iir=False, win_t=10, step_t=5, win_type=1):
    # UI接口对接，浮点数转整型
    N = int(N)
    batch_size = int(batch_size)

    # 加载音频
    # input_path1 无噪，input_path2 有噪
    format1 = input_path1[-3:]
    format2 = input_path2[-3:]
    music1, music2 = None, None
    if format1 == 'wav':
        music1 = AudioSegment.from_wav(input_path1)
    elif format1 == 'mp3':
        music1 = AudioSegment.from_mp3(input_path1)
    else:
        print("File1: Unsupported audio file type!")
        return

    if format2 == 'wav':
        music2 = AudioSegment.from_wav(input_path2)
    elif format2 == 'mp3':
        music2 = AudioSegment.from_mp3(input_path2)
    else:
        print("File2: Unsupported audio file type!")
        return

    fs, fw, channels = music1.frame_rate, music1.frame_width, music1.channels
    data = np.asarray(music1.get_array_of_samples(), float).reshape(-1, channels)
    data2 = np.asarray(music2.get_array_of_samples(), float).reshape(-1, channels)

    # 关联度检测（防止误传文件）
    if len(data) != len(data2):
        print("Unmatched music with noise and without noise!")
        return

    # 单声道训练，双声道应用
    if filter is None:
        # 注意不要学习a0（a0和其他ai系数很接近，因此除以a0之后容易发生迭代式的数值爆炸）
        # 学习a1-aN同样会产生数值爆炸(大部分abs(ai)<<1)，这也可能造成往往不具有参数数值可解释性的
        # 深度学习/机器学习在一些需要明确参数意义的任务上效果不是很好
        a_list = np.zeros(N)
        b_list = (2 / N + 1) ** 0.5 * np.random.randn(N + 1)

        if iir:
            a_list = (2 / N + 1) ** 0.5 * np.random.randn(N + 1)

        # 长度检测，FIR/IIR默认Mini-Batch批大小为32*(N+1)(N+1是FIR滤波器点数，IIR点数2N+1)
        # 如果有剩余不足batch_l的片段也单独进行训练
        batch_l = (N + 1) * batch_size

        if len(data) < batch_l:
            print(f"Not enough samples!\nSamples:{len(data)} Batch_length:{batch_l}")
            return

        # 多声道合并-归一化数据预处理，手动梯度下降（因为反向梯度传播较为简单，不需要调用库）
        temp, temp2 = np.zeros((len(data))), np.zeros(len(data))
        for i in range(channels):
            temp[:] += data[:, i]
            temp2[:] += data2[:, i]

        temp /= channels
        temp2 /= channels
        data = temp / np.max(abs(temp))
        data2 = temp2 / np.max(abs(temp2))

        train_loss = []

        # for epoch in range(Epochs):
        epoch = 0
        if iir:
            while True:
                # 不使用余弦衰减型学习率，因为训练总轮数Epochs不确定，训练结束的标志是当前轮epoch的l2损失小于给定thres
                # alpha=alpha_initial*(0.5+np.cos(epoch/Epochs))
                # alpha = alpha_initial * np.exp2(-epoch / 100)
                alpha = alpha_initial
                l2 = 0
                a_fir_momentum, b_fir_momentum = 0, 0

                for i in range(0, len(data) - len(data) % (N + 1), batch_l):
                    grad_a, grad_b = 0, 0
                    # Mini-Batch 逻辑(批与批之间不重叠，独立训练)
                    # 如果有剩余不足batch_l的片段也单独进行训练,但是也需要确保单个实例得以保全
                    for j in range(0, min(batch_l, len(data) - len(data) % (N + 1) - i), N + 1):
                        grad_a += 2 * (np.dot(a_list, data[i + j:i + j + N]) + data[i + j + N]
                                       - np.dot(b_list, data2[i + j:i + j + N + 1])) * data[i + j:i + j + N + 1]
                        grad_b -= 2 * (np.dot(a_list, data[i + j:i + j + N]) + data[i + j + N]
                                       - np.dot(b_list, data2[i + j:i + j + N + 1])) * data2[i + j:i + j + N + 1]
                        l2 += ((np.dot(a_list, data[i + j:i + j + N]) + data[i + j + N]
                                - np.dot(b_list, data2[i + j:i + j + N + 1]))) ** 2

                    # Parameters Update
                    # L2 regularization
                    if method == 'sgd':
                        a_list -= alpha * (grad_a / min(batch_size, (len(data) - len(data) % (N + 1) - i) // (
                                N + 1)) + 2 * 1e-5 * a_list)
                        b_list -= alpha * (grad_b / min(batch_size, (len(data) - len(data) % (N + 1) - i) // (
                                N + 1)) + 2 * 1e-5 * b_list)
                    elif method == 'sgd-momentum':
                        a_fir_momentum = rho * a_fir_momentum + (1 - rho) * grad_a
                        b_fir_momentum = rho * b_fir_momentum + (1 - rho) * grad_b
                        a_list -= alpha * (a_fir_momentum / min(batch_size, (len(data) - len(data) % (N + 1) - i) // (
                                N + 1)) + 2 * 1e-5 * a_list)
                        b_list -= alpha * (b_fir_momentum / min(batch_size, (len(data) - len(data) % (N + 1) - i) // (
                                N + 1)) + 2 * 1e-5 * b_list)
                train_loss.append(l2)
                epoch += 1
                print(f"Epoch:{epoch + 1},loss:{l2}")
                if l2 < len(data) // (2 * N + 2) * thres ** 2:
                    break

        else:
            while True:
                # alpha=alpha_initial*(0.5+np.cos(epoch/Epochs))
                # alpha = alpha_initial * np.exp2(-epoch / 100)
                alpha = alpha_initial
                l2 = 0
                first_mom = 0
                for i in range(0, len(data) - len(data) % (N + 1), batch_l):
                    grad_b = 0

                    # Mini-Batch 逻辑(批与批之间不重叠，独立训练)
                    # 如果有剩余不足batch_l的片段也单独进行训练,但是也需要确保单个实例得以保全
                    for j in range(0, min(batch_l, len(data) - len(data) % (N + 1) - i), N + 1):
                        grad_b -= 2 * (data[i + j + N] - np.dot(b_list, data2[i + j:i + j + N + 1])) * data2[
                                                                                                       i + j:i + j + N + 1]
                        l2 += (data[i + j + N] - np.dot(b_list, data2[i + j:i + j + N + 1])) ** 2

                    # Parameters Update
                    # L2 regularization
                    if method == 'sgd':
                        b_list -= alpha * (grad_b / min(batch_size, (len(data) - len(data) % (N + 1) - i) // (
                                    N + 1)) + 2 * 1e-5 * b_list)
                    elif method == 'sgd-momentum':
                        first_mom = rho * first_mom + (1 - rho) * grad_b
                        b_list -= alpha * (first_mom / min(batch_size, (len(data) - len(data) % (N + 1) - i) // (
                                    N + 1)) + 2 * 1e-5 * b_list)

                train_loss.append(l2)
                epoch += 1
                print(f"Epoch:{epoch + 1},loss:{l2}")
                if l2 < len(data) // (N + 1) * thres ** 2:
                    break

        # 学习曲线可视化
        plt.figure()
        plt.plot(range(1, len(train_loss) + 1), train_loss, 'b-o')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('L2 loss')
        plt.title(f'Learning Curve N={N} FIR:{not iir} IIR:{iir}')
        plt.show()

        print(f"FIR:{not iir} IIR:{iir} Learned coefficients:\na:{a_list}\nb:{b_list}")
        return b_list, a_list, train_loss

    # filter不是空，则代表evaluation/prediction
    else:
        b_list, a_list = filter
        # 此处N的值最好从filter参数获取
        N = len(b_list) - 1

        # 学习参数不包括a0效果可能更好
        # b_list/=a_list[0]
        # a_list=a_list[1:]/a_list[0]

        # input_path2/data2对应噪声信号
        # 严格来说应该按照声道分别归一化，此处直接多声道统一处理（差别应该不大，未试验）
        data2 /= np.max(abs(data2))

        # 计算前前面补零，计算完成后切片去头
        # 和训练时不同，data2 一定也要对齐补零，涉及到非因果变成因果的N点时移
        # （虽然数据操作看起来没有大问题，但是从去噪效果和物理意义上来说差别不小）
        data2 = np.concatenate((np.zeros((N, channels)), data2), axis=0)
        data3 = np.zeros((len(data2), channels), np.float64)

        # 在时域通过向量内积实现IIR滤波，注意整个过程都是多声道进行
        for i in range(N, len(data)):
            data3[i] = np.dot(b_list, data2[i - N:i + 1]) - np.dot(a_list, data3[i - N:i])

        # 逆缩放同样可以采用手动梯度下降（L2差异最小化，data3_scaled和data2_orig）
        # 绝对音量差异不关键
        # 确保参数具备一定的物理意义（防止数值爆炸这种数据无效的情况）
        print(np.max(abs(data3)))
        # assert np.max(abs(data3))<=2
        data3 = data3[N:]
        data2 = data2[N:]

        # 音频处理与导出

        # 16-bit quantization
        if fw / channels == 2:
            # 最大音量调整到支持最大音量的70%
            data3 = np.asarray(data3 / np.max(abs(data3)) * 2 ** 15 * 0.7, np.int16)
        else:
            # 尽管24比特量化精度未达到32，但是放缩处理只改变了步长<->绝对精度，不会改变相对精度，相当于音量拉低拉高
            # 导出的时候还是分开导出
            data3 = np.asarray(data3 / np.max(abs(data3)) * 2 ** 31 * 0.7, np.int32)

        assert (len(data3) == len(data))
        # STFT与语谱图矩阵计算
        M2 = STFT(data3, fs, win_t, step_t, win_type)
        M2 = (abs(M2)[:, :len(M2[0]) // 2 + 1]).T
        M2 = np.asarray(np.minimum(M2 / np.mean(M2) * 128, 255 * np.ones_like(M2)), np.uint8)

        M = STFT(data, fs, win_t, step_t, win_type)
        M = (abs(M)[:, :len(M[0]) // 2 + 1]).T
        M = np.asarray(np.minimum(M / np.mean(M) * 128, 255 * np.ones_like(M)), np.uint8)

        img_ss(M, M2, np.ones_like(M), 0, fs / 2, fs, win_t, step_t)

        # 支持同目录导出
        base_l = len(os.path.basename(input_path2))
        filename = os.path.basename(input_path2)[:-4]
        dir = input_path2[:-base_l]
        output_path = dir + filename + '_saf.mp3'
        print(f'dir:{dir}\nfilename:{filename}\noutput_filepath:{output_path}\n')
        export_audio = AudioSegment(data=data3.tobytes(), frame_rate=fs, sample_width=int(fw / channels),
                                    channels=channels)
        export_audio.export(output_path, 'mp3')

        return output_path, M2


def get_fir_coeff(fir, N, fl, fh, dir, mpath, nmpath):
    b_list = None
    a_list = np.zeros(N)
    if not fir:
        if f'freq{fl}-{fh}__IIRb_SAF1.csv' not in os.listdir(dir):
            b_list, a_list, train_loss = filter_self_adaptive(mpath, nmpath, N, 32, 0.01, 0.001, 'sgd-momentum', 0.9,
                                                              iir=True)
            np.savetxt(f'freq{fl}-{fh}__IIRb_SAF1.csv', b_list)
            np.savetxt(f'freq{fl}-{fh}__IIRa_SAF1.csv', a_list)
        else:
            b_list = np.loadtxt(f'freq{fl}-{fh}__IIRb_SAF1.csv')
            a_list = np.loadtxt(f'freq{fl}-{fh}__IIRa_SAF1.csv')
    else:
        if f'freq{fl}-{fh}__FIR_SAF2.csv' not in os.listdir(dir):
            b_list, a_list, train_loss = filter_self_adaptive(mpath, nmpath, 2 * N + 1, 32, 0.01, 0.001, 'sgd-momentum',
                                                              0.9, iir=False)
            np.savetxt(f'freq{fl}-{fh}__FIR_SAF2.csv', b_list)
        else:
            b_list = np.loadtxt(f'freq{fl}-{fh}__FIR_SAF2.csv')

    return b_list, a_list


if __name__ == "__main__":
    # 执行分离
    # separate_vocals(input_path=r"d:\wf200\Music\mixed\Counting_Stars_44k_16.mp3",output_dir=r"d:\wf200\Music\output",model_type="2stems")
    fl = 15000
    fh = 16000
    win_t = 20
    step_t = 10
    win_t2 = 200
    win_type = 1
    orig_path = r"d:\wf200\Music\test\watch_snow.mp3"
    orig_dir = r"d:\wf200\Documents\mystudy\大二下\数字信号处理\DSP实验\HW4"
    # R(order) calculated by fs/(1000/t_delay), corresponding frequency precision:1000/t_delay
    # R:2205<->t_delay:50ms<->f_prec:20Hz

    low_order = 5
    r_t = 400
    N = 2999
    # echoing(orig_path,0.5,0.5,0.5,200,200,200,region_t=r_t,low_order=low_order)
    # eq_path=equalizer(orig_path,(0,0,-20,-30,-30,0,40,40,0,0),2**(0.5),4410,False,100)

    # mpath,npath,nmpath,M=addnoise(orig_path,16000,1000,1,54,18,fl,fh,-1,win_t,step_t,1)
    mpath = r"d:\wf200\Music\test\music_seg(54-72).mp3"
    nmpath = r"d:\wf200\Music\test\noisemusic(54-72_1_f15000_f16000_16000_1000_0).mp3"
    path2, M2 = filter(mpath, nmpath, fl, fh, win_t, step_t, win_t2, 1, False)
    # path3,M3=filter(mpath,nmpath,fl,fh,win_t,step_t,win_t2,1,True)
    FIR = True
    # b_list,a_list=get_fir_coeff(FIR,N,fl,fh,orig_dir,mpath,nmpath)
    # print(f"Training loss:{train_loss}")
    # path4,M4=filter_self_adaptive(mpath,nmpath,filter=(b_list,a_list),win_t=win_t,step_t=step_t,win_type=win_type,iir=not FIR)

    # plt.show()

