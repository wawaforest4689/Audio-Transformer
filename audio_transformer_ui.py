"""
2025/06/21
Kimi Assistant
代码说明：
导入必要的库：导入PyQt5库中的相关模块。
创建主窗口类：定义一个EqualizerUI类，继承自QWidget。
初始化界面：在initUI方法中，创建主布局并分为左右两个区域。
左侧区域（文件上传）：使用QPushButton和QFileDialog实现文件上传功能。
右侧区域（均衡器）：使用QSlider和QLabel创建均衡器滑块和标签。
滑块值变化处理：通过sliderValueChanged方法处理滑块值变化，并更新增益值。
获取增益值：通过getGains方法获取当前的增益值。
"""
import numpy as np

from backend_operations import *
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from pydub import AudioSegment


class EqualizerUI(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.gains = [0] * 10  # 初始化增益数组
        self.audioFiles = []  # 存储导入的音频文件
        self.audioFilesOut = []  # 存储导出的音频文件
        self.isExported = False  # 导出状态
        # two elements storage for combination function and FIR/Self-adaptive filtering
        self.audio_name=["",""]

        self.Q=2**0.5
        self.N=512
        self.eq_et=400
        self.echo_et=400
        self.echo_men=5
        self.echo_udelays=[100,100,100]
        self.echo_alphas=[0.5,0.5,0.5]
        self.noise_fl=40
        self.noise_fh=60
        self.noise_ty=1
        self.noise_avg=16000
        self.noise_std=800
        self.noise_beg=0
        self.noise_dur=30
        self.noise_step=-1
        self.noise_fwt=200
        self.fft_wti=10
        self.fft_st=5
        self.fft_wty=1

        self.safilter=None



    def initUI(self):
        # 创建主布局
        mainLayout = QHBoxLayout()

        # 创建左侧布局（文件上传和导出）
        leftLayout = QVBoxLayout()

        # 上半区域（导入文件列表）
        self.importLayout = QVBoxLayout()
        self.uploadButton = QPushButton('Add More Files')
        self.uploadButton.clicked.connect(self.uploadFiles)
        self.importLayout.addWidget(self.uploadButton)
        self.fileListImport = QListWidget()
        self.fileListImport.itemDoubleClicked.connect(self.playAudio)
        self.fileListImport.setSortingEnabled(True)
        # self.fileListImport.setDragEnabled(True)
        # No connected attribute!
        # self.fileListImport.itemClicked.connected(self.audiochose)
        self.importLayout.addWidget(self.fileListImport)

        leftLayout.addLayout(self.importLayout)

        # 中间区域（导出文件列表）
        self.exportLayout2=QVBoxLayout()
        self.fileListExport = QListWidget()
        self.fileListExport.itemDoubleClicked.connect(self.playAudio)
        self.fileListExport.setSortingEnabled(True)
        self.exportLayout2.addWidget(self.fileListExport)

        leftLayout.addLayout(self.exportLayout2)

        
        # 下半区域（按钮与部分参数设置）
        self.exportwidget=QGroupBox()
        self.exportLayout = QGridLayout(self.exportwidget)

        self.exportLayout.addWidget(QLabel("Output directory:"),0,0)
        self.output_dir=QLineEdit("")
        self.exportLayout.addWidget(self.output_dir,0,1)


        self.Echo_saveButton = QPushButton('Echo-save')
        self.Echo_saveButton.clicked.connect(self.Echo_saveAudio)
        self.exportLayout.addWidget(self.Echo_saveButton,1,0)


        self.EQ_saveButton = QPushButton('EQ-save')
        self.EQ_saveButton.clicked.connect(self.EQ_saveAudio)
        self.exportLayout.addWidget(self.EQ_saveButton,1,1)


        self.noise_saveButton = QPushButton('Noise-save')
        self.noise_saveButton.clicked.connect(self.noise_saveAudio)
        self.exportLayout.addWidget(self.noise_saveButton,2,0)
        
        
        self.filter_saveButton = QPushButton('Filter-save')
        self.filter_saveButton.clicked.connect(self.filter_saveAudio)
        self.exportLayout.addWidget(self.filter_saveButton,2,1)
        
        self.comb_saveButton = QPushButton('Combine-save')
        self.comb_saveButton.clicked.connect(self.comb_saveAudio)
        self.exportLayout.addWidget(self.comb_saveButton,3,0)
        
        self.split_saveButton = QPushButton('Split-save')
        self.split_saveButton.clicked.connect(self.split_saveAudio)
        self.exportLayout.addWidget(self.split_saveButton,3,1)

        self.rs_saveButton = QPushButton('Remove-silence-save')
        self.rs_saveButton.clicked.connect(self.rs_saveAudio)
        self.exportLayout.addWidget(self.rs_saveButton,4,0)

        self.saf_saveButton = QPushButton('Self-adaptive-filter-save')
        self.saf_saveButton.clicked.connect(self.saf_saveAudio)
        self.exportLayout.addWidget(self.saf_saveButton,4,1)

        # self.exportLayout.addWidget(QLabel("Train:"),5,0)
        self.saf_checkbox = QCheckBox("(Train)")
        self.saf_checkbox.setChecked(True)
        self.exportLayout.addWidget(self.saf_checkbox,5,0)

        self.exportLayout.addWidget(QLabel("(N,batch_size,alpha0,thres):"), 6, 0)
        self.saf_ps = QLineEdit("3999,32,0.1,0.001")
        self.exportLayout.addWidget(self.saf_ps, 6, 1)


        leftLayout.addWidget(self.exportwidget)

        # 加噪-去噪模块
        noise=QGroupBox("noise addition and removal")
        noise_layout=QGridLayout(noise)

        noise_layout.addWidget(QLabel("noise settings(freql,freqh,avg,std,begin,duration,step):"),0,0)
        self.noise_params=QLineEdit("40,60,16000,800,0,30,-1")
        noise_layout.addWidget(self.noise_params)

        noise_layout.addWidget(QLabel("noise type:"),0,1)
        self.noise_type=QComboBox()
        self.noise_type.addItems(['gaussian','uniform'])
        self.noise_type.currentIndexChanged.connect(self.change_noise_type)
        noise_layout.addWidget(self.noise_type,0,2)

        noise_layout.addWidget(QLabel("filtering window time(ms):"),2,0)
        self.noise_filtert=QLineEdit("200")
        noise_layout.addWidget(self.noise_filtert,2,1)

        leftLayout.addWidget(noise)

        # 窗函数模块（语谱图显示）
        window=QGroupBox("STFT Window settings")
        window_layout=QGridLayout(window)

        window_layout.addWidget(QLabel("Window time:"),0,0)
        self.window_time=QComboBox()
        self.window_time.addItems(['10ms','20ms','40ms','50ms','100ms','200ms','250ms','500ms','1s'])
        self.window_time.currentIndexChanged.connect(self.change_win_time)
        window_layout.addWidget(self.window_time,0,1)

        window_layout.addWidget(QLabel("Step time(ms):"),1,0)
        self.window_stime=QLineEdit("5")
        window_layout.addWidget(self.window_stime,1,1)

        window_layout.addWidget(QLabel("Type:"),2,0)
        self.window_type=QComboBox()
        self.window_type.addItems(['Rectangular window','Hanning window','Hamming window','Blackman window'])
        self.window_type.currentIndexChanged.connect(self.change_win_type)
        window_layout.addWidget(self.window_type,2,1)

        leftLayout.addWidget(window)

        mainLayout.addLayout(leftLayout)


        # 创建右侧布局（均衡器）
        rightLayout = QVBoxLayout()
        self.sliders = []
        self.labels = []
        frequencies = ['31', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '16k']
        for i, freq in enumerate(frequencies):
            gridLayout = QGridLayout()
            label = QLabel(freq+'Hz')
            gridLayout.addWidget(label, 1, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-50)
            slider.setMaximum(50)
            slider.setValue(0)
            slider.setSingleStep(1)
            slider.setTickInterval(5)
            slider.setTickPosition(QSlider.TicksAbove)
            slider.valueChanged.connect(self.sliderValueChanged)
            gridLayout.addWidget(slider, 1, 1)


            # 添加刻度标签
            tick_label=""
            for j in range(-50, 51, 5):
                tick_label+=str(j)+"dB  "
            tick_label = QLabel(tick_label)
            tick_label.setAlignment(Qt.AlignCenter)
            tick_label.resize(slider.width(),tick_label.height())
            gridLayout.addWidget(tick_label, 0, 1)

            self.sliders.append(slider)
            self.labels.append(label)
            rightLayout.addLayout(gridLayout)


        # 均衡器模块
        EQ=QGroupBox("EQ")
        EQ_layout=QGridLayout(EQ)

        Nlabel=QLabel("Length of filter:")
        EQ_layout.addWidget(Nlabel,0,0)
        self.eq_Ninput=QLineEdit("512")
        EQ_layout.addWidget(self.eq_Ninput,0,1)

        quality_f_label=QLabel("Quality factor([1,inf)):")
        EQ_layout.addWidget(quality_f_label,1,0)
        self.eq_qfinput=QLineEdit("1.4142")
        EQ_layout.addWidget(self.eq_qfinput,1,1)

        EQ_layout.addWidget(QLabel("EQ effective time:"),2,0)
        self.eq_effectivetime=QComboBox()
        self.eq_effectivetime.addItems(["100ms","150ms","200ms","250ms","300ms","350ms","400ms","450ms","500ms"])
        self.eq_effectivetime.currentIndexChanged.connect(self.change_eq_et)
        EQ_layout.addWidget(self.eq_effectivetime,2,1)

        rightLayout.addWidget(EQ)


        # 回声模块
        echo=QGroupBox("Echoing")
        echo_layout=QGridLayout(echo)

        echo_layout.addWidget(QLabel("Effective time:"),0,0)
        self.echo_effectivetime=QComboBox()
        self.echo_effectivetime.addItems(["100ms","150ms","200ms","250ms","300ms","350ms","400ms","450ms","500ms"])
        self.echo_effectivetime.currentIndexChanged.connect(self.change_echo_et)
        echo_layout.addWidget(self.echo_effectivetime,0,1)

        echo_layout.addWidget(QLabel("Maximum_echo_number:"),1,0)
        self.echo_minnumber=QSpinBox()
        self.echo_minnumber.setRange(2,100)
        self.echo_minnumber.setSingleStep(1)
        echo_layout.addWidget(self.echo_minnumber,1,1)

        echo_layout.addWidget(QLabel("Unit delay(ms,(t1,t2,t3)):"),2,0)
        self.echo_ud=QLineEdit("50 50 50")
        echo_layout.addWidget(self.echo_ud,2,1)

        echo_layout.addWidget(QLabel("Unit delay(ms):"),2,0)
        self.echo_ud=QLineEdit("50")
        echo_layout.addWidget(self.echo_ud,2,1)

        echo_layout.addWidget(QLabel("Echoing coefficient(a1,a2,a3):"),3,0)
        self.echo_coeff=QLineEdit("0.5 0.5 0.5")
        echo_layout.addWidget(self.echo_coeff,3,1)

        rightLayout.addWidget(echo)

        mainLayout.addLayout(rightLayout)


        self.setLayout(mainLayout)
        self.setWindowTitle('Audio Transformer')
        self.setGeometry(50, 50, 1200, 200)


    def uploadFiles(self):
        options = QFileDialog.Options()
        fileNames, _ = QFileDialog.getOpenFileNames(self, "Choose Files", "", "Audio Files (*.ogg *.mp3);All Files (*);", options=options)
        if fileNames:
            for fileName in fileNames:
                self.audioFiles.append(fileName)
                item = QListWidgetItem(fileName)
                self.fileListImport.addItem(item)

    def sliderValueChanged(self):
        self.gains = [slider.value() for slider in self.sliders]

    def playAudio(self, item=None):
        if item:
            if self.audio_name[0] == "":
                self.audio_name[0] = item.text()
            else:
                self.audio_name[1] = self.audio_name[0]
                self.audio_name[0] = item.text()
            print(f"Playing audio: {self.audio_name[0]}")
            play_music(self.audio_name[0])


    def EQ_saveAudio(self):
        if self.audio_name[0]!="":
            # 参数更新
            self.Q=float(self.eq_qfinput.text())
            self.N=int(self.eq_Ninput.text())
            print(f"parameters:{self.audio_name[0],self.gains,self.Q,self.N,True,self.eq_et}\n")

            exportedFileName=equalizer(self.audio_name[0],self.gains,self.Q,self.N,True,self.eq_et)
            print(f"Saving audio: {exportedFileName}")

            if exportedFileName not in self.audioFilesOut:
                self.audioFilesOut.append(exportedFileName)
                item = QListWidgetItem(exportedFileName)
                self.fileListExport.addItem(item)

            self.isExported = True
            self.audio_name[0]=""
            self.audio_name[1]=""

    def Echo_saveAudio(self):
        if self.audio_name[0]!="":
            self.echo_men = self.echo_minnumber.value()
            print(int(s) for s in self.echo_ud.text().split(',',-1))
            self.echo_udelays = tuple(int(s) for s in self.echo_ud.text().split(',',-1))
            self.echo_alphas = tuple(float(s) for s in self.echo_coeff.text().split(',',-1))
            self.fft_st=int(self.window_stime.text())
            assert(self.fft_st<self.fft_wti)

            comb_p,scomb_p,allp_p,_,_=echoing(self.audio_name[0],*self.echo_alphas,*self.echo_udelays,self.fft_wti,self.fft_st,
                    self.fft_wty,True,True,self.echo_et,self.echo_men)

            print(f"Saving audios: {comb_p}\n{scomb_p}\n{allp_p}\n")

            if comb_p not in self.audioFilesOut:
                self.audioFilesOut.append(comb_p)
                item = QListWidgetItem(comb_p)
                self.fileListExport.addItem(item)

            if scomb_p not in self.audioFilesOut:
                self.audioFilesOut.append(scomb_p)
                item = QListWidgetItem(scomb_p)
                self.fileListExport.addItem(item)

            if allp_p not in self.audioFilesOut:
                self.audioFilesOut.append(allp_p)
                item = QListWidgetItem(allp_p)
                self.fileListExport.addItem(item)

            self.audio_name[0]=""
            self.audio_name[1]=""

    def noise_saveAudio(self):
        if self.audio_name[0]!="":
            print(self.audio_name[0])
            print(self.noise_params.text().split(',', -1))
            print((int(s) for s in self.noise_params.text().split(',', -1)))
            params = tuple(int(s) for s in self.noise_params.text().split(',', -1))
            self.noise_fl = params[0]
            self.noise_fh = params[1]
            self.noise_avg = params[2]
            self.noise_std = params[3]
            self.noise_beg = params[4]
            self.noise_dur = params[5]
            self.noise_step = params[6]
            m_path, n_path, nm_path, _ = addnoise(self.audio_name[0], self.noise_avg, self.noise_std, self.noise_ty,
                                           self.noise_beg, self.noise_dur,
                                           self.noise_fl, self.noise_fh, self.noise_step, self.fft_wti, self.fft_st,
                                           self.fft_wty)
            print(f"Saving audio: {m_path}\n{n_path}\n{nm_path}\n")

            if m_path not in self.audioFilesOut:
                self.audioFilesOut.append(m_path)
                item = QListWidgetItem(m_path)
                self.fileListExport.addItem(item)

            if n_path not in self.audioFilesOut:
                self.audioFilesOut.append(n_path)
                item = QListWidgetItem(n_path)
                self.fileListExport.addItem(item)

            if nm_path not in self.audioFilesOut:
                self.audioFilesOut.append(nm_path)
                item = QListWidgetItem(nm_path)
                self.fileListExport.addItem(item)

            self.isExported = True
            self.audio_name[0]=""
            # 为避免混乱，直接清空所有存储(考虑可视化)
            self.audio_name[1]=""


    def filter_saveAudio(self):
        if self.audio_name[0]!="" and self.audio_name[1]!="":
            print(int(s) for s in self.noise_params.text().split(',',-1))
            params=tuple(int(s) for s in self.noise_params.text().split(',',-1))
            self.noise_fl = params[0]
            self.noise_fh = params[1]
            self.noise_fwt = params[8]

            filter_path,_=filter(self.audio_name[1],self.audio_name[0],self.noise_fl,self.noise_fh,self.fft_wti,self.fft_st,
                   self.noise_fwt,self.fft_wty,True,True)
            print(f"Saving audio: {filter_path}")
            if filter_path not in self.audioFilesOut:
                self.audioFilesOut.append(filter_path)
                item = QListWidgetItem(filter_path)
                self.fileListExport.addItem(item)

            self.isExported = True
            self.audio_name[0]=""
            self.audio_name[1]=""


    def comb_saveAudio(self):
        if self.audio_name[0]!="" and self.audio_name[1]!="":
            output_dir=self.output_dir.text()
            file_path=comb_music(self.audio_name[0],self.audio_name[1],output_dir)

            print(f"Saving audio: {file_path}")
            if file_path not in self.audioFilesOut:
                self.audioFilesOut.append(file_path)
                item = QListWidgetItem(file_path)
                self.fileListExport.addItem(item)

            self.isExported = True

            self.audio_name[0]=""
            self.audio_name[1]=""




    def split_saveAudio(self):
        if self.audio_name[0]!="":
            output_dir=self.output_dir.text()
            vocal,bgm=separate_vocals(input_path=self.audio_name[0],output_dir=output_dir)
            print(f"Saving audios: {vocal}\n{bgm}\n")
            if vocal not in self.audioFilesOut:
                self.audioFilesOut.append(vocal)
                item = QListWidgetItem(vocal)
                self.fileListExport.addItem(item)

            if bgm not in self.audioFilesOut:
                self.audioFilesOut.append(bgm)
                item = QListWidgetItem(bgm)
                self.fileListExport.addItem(item)

            self.isExported = True
            self.audio_name[0]=""
            self.audio_name[1]=""
            
    def rs_saveAudio(self):
        if self.audio_name[0]!="":
            output_path=remove_silence_ui(self.audio_name[0])
            print(f"Saving audios: {output_path}\n")
            if output_path not in self.audioFilesOut:
                self.audioFilesOut.append(output_path)
                item = QListWidgetItem(output_path)
                self.fileListExport.addItem(item)

            self.isExported = True
            self.audio_name[0]=""
            self.audio_name[1]=""

    def saf_saveAudio(self):
        if self.audio_name[0]!="" and self.audio_name[1]!="":
            if self.saf_checkbox.isChecked():
                saf_ps=tuple(float(s) for s in self.saf_ps.text().split(",",-1))
                N,b_s,a0,thres=saf_ps
                # 先双击无噪，后双击有噪
                b_list,a_list=filter_self_adaptive(self.audio_name[1],self.audio_name[0],N,b_s,a0,thres)
                self.safilter=(b_list,a_list)
                print("Trained successfully!")

            elif self.safilter is not None:
                output_path=filter_self_adaptive(self.audio_name[1],self.audio_name[0],filter=self.safilter)
                print(f"Saving audios: {output_path}\n")
                if output_path not in self.audioFilesOut:
                    self.audioFilesOut.append(output_path)
                    item = QListWidgetItem(output_path)
                    self.fileListExport.addItem(item)

                self.isExported = True
                self.audio_name[0] = ""
                self.audio_name[1] = ""



    def audiochose(self,item=None):
        if item:
            if self.audio_name[0]=="":
                self.audio_name[0]=item.text()
            else:
                self.audio_name[1]=self.audio_name[0]
                self.audio_name[0]=item.text()

    def change_eq_et(self):
        eq_ets=[100,150,200,250,300,350,400,450,500]
        self.eq_et=eq_ets[self.eq_effectivetime.currentIndex()]
    def change_echo_et(self):
        eq_ets=[100,150,200,250,300,350,400,450,500]
        self.echo_et=eq_ets[self.echo_effectivetime.currentIndex()]
    def change_noise_type(self):
        self.noise_ty=self.noise_type.currentIndex() + 1

    def change_win_time(self):
        win_times=[10,20,40,50,100,200,250,500,1000]
        self.fft_wti=win_times[self.window_time.currentIndex()]
    def change_win_type(self):
        self.fft_wty=self.window_type.currentIndex()+1

    def keyPressEvent(self, a0,QkeyPressEvent=None):
        if a0.key()==Qt.Key.Key_Delete:
            print(self.fileListImport.selectedItems())
            if self.fileListImport.selectedItems():
                index=self.fileListImport.currentRow()
                print(index)
                if self.audio_name[0]==self.audioFiles[index]:
                    # for item in self.fileListImport.selectedItems():
                        # del item
                    # item_ptr = self.fileListImport.currentItem()
                    # print(item_ptr)
                    self.fileListImport.takeItem(index)
                    # del item_ptr
                    # self.fileListImport.clear()
                    self.audioFiles.pop(index)

            if self.fileListExport.selectedItems():

                index=self.fileListExport.currentRow()
                print(index)
                if self.audio_name[0]==self.audioFilesOut[index]:
                    # for item in self.fileListExport.selectedItems():
                        # del item
                    # item_ptr = self.fileListExport.currentItem()
                    self.fileListExport.takeItem(index)
                    self.audioFilesOut.pop(index)


if __name__ == '__main__':

    music_path=r"d:\wf200\Music\test\Wake(Live).mp3"
    # play_music(music_path)

    app = QApplication(sys.argv)
    ex = EqualizerUI()
    ex.show()
    sys.exit(app.exec_())


