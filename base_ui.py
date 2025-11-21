import sys
import torch
import cv2
import os
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PySide6.QtGui import QPixmap,QImage
from PySide6.QtCore import QTimer, QThread, Signal, Slot, Qt, QObject
from main_window_ui import Ui_MainWindow  # Import the generated UI class

# 模型加载线程
class ModelLoadingThread(QThread):

    # 修改信号定义：加载成功时只发射一个布尔值 (True)
    model_loaded = Signal(bool)

    # 用户定义的加载失败信号 (保持原样)
    model_load_error = Signal(str)

    def __init__(self, repo_path, absolute_weights_path, parent=None):
        super().__init__(parent)
        self.repo_path = repo_path
        self.absolute_weights_path = absolute_weights_path
        self._is_running = True

    def run(self):
        try:
            # 在线程中加载模型以验证路径和权重是否有效
            # 注意：加载完成后我们不在这里存储模型，也不发射模型对象
            model_test_load = torch.hub.load(
                self.repo_path,
                'custom', # 你的 hubconf.py 入口点
                path=self.absolute_weights_path, # 传递权重路径
                source='local',
                force_reload=True 
            )
            # 如果加载测试成功，发射成功信号 (只发送 True)
            if self._is_running:
                 self.model_loaded.emit(True) 
        except Exception as e:
            # 如果加载失败，发射错误信号 (保持原样)
            if self._is_running:
                 self.model_load_error.emit(f"模型加载失败: {e}") 

    def stop(self):
        """Provides a way to signal the thread to stop."""
        self._is_running = False

# 图片检测线程
class ImageDetectionThread(QThread):

    # 信号量：当检测完成时发出信号，携带检测后的图片
    detection_finished = Signal(np.ndarray)

    # 信号量：检测过程中出现错误，携带错误信息
    detection_error = Signal(str)

    def __init__(self, model, image_path, parent = None):
        super().__init__(parent)
        self.model = model
        self.image_path = image_path
        self._is_running = True  # 控制线程运行的标志

    def run(self):
        if self.model is None:
            if self._is_running:
                self.detection_error.emit("Model is not loaded.")
            return
        try:
            results = self.model(self.image_path)  # Perform detection
            if not self._is_running:
                return
            render_results = results.render()  # Render results
            if (render_results == 0):
                if self._is_running:
                    self.detection_error.emit("No results to render.")
                return
            annotated_image_np = render_results[0]  # Get the first rendered image
            if not annotated_image_np.flags.writeable:
                annotated_image_np = np.copy(annotated_image_np)
            if self._is_running:
                self.detection_finished.emit(annotated_image_np)
        except Exception as e:
            if self._is_running:
                self.detection_error.emit(f"Error during detection: {str(e)}")
    def stop(self):
        self._is_running = False

# 视频检测工作者线程
class VideoDetectionWorker(QObject):

    # 信号量，当单帧检测完成时发射，携带检测后的图像数据（一个np数组）
    detection_finished = Signal(np.ndarray)

    # 信号量，当检测过程中出现错误时发射，携带错误信息
    detection_error = Signal(str)
    
    def __init__(self, model, parent =None):
        super().__init__(parent) 
        self.model = model
        self._is_running = True  # 控制线程运行的标志

    # slot function for recieving the frame from the video
    @Slot(np.ndarray)
    def process_frame(self, frame_bgr_np):
        """接收一帧 BGR 格式的视频数据，进行检测，并发射结果。"""
        if not self._is_running:
            return
        if self.model is None:
            if self._is_running:
                self.detection_error.emit("Model is not loaded.")
            return
        if frame_bgr_np is None or not isinstance(frame_bgr_np, np.ndarray):
            if self._is_running:
                self.detection_error.emit("Received empty frame.")
            return
        try:
            results = self.model(frame_bgr_np)  # Perform detection
            render_results = results.render()  # Render results
            if (len(render_results) == 0):
                if self._is_running:
                    self.detection_error.emit("No results to render.")
                return
            annotated_frame_np = render_results[0]  # Get the first rendered image
            if not annotated_frame_np.flags.writeable:
                annotated_frame_np = np.copy(annotated_frame_np)  # Get the first rendered image
            # 发射信号，将处理好的结果返回主线程
            if self._is_running:
                self.detection_finished.emit(annotated_frame_np)
        except Exception as e:
            if self._is_running:
                pass

    def stop(self):
        self._is_running = False

def convert2img(img):

    height, width, channel = img.shape
    return QImage(img.data, width, height, channel * width, QImage.Format_RGB888)


class MainWindow(QMainWindow, Ui_MainWindow):

    # Signal to communicate between threads
    process_video_frame_signal = Signal(np.ndarray)  # Signal to process video frame

    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Set up the UI from the generated class
        self.setWindowTitle("Fire and Smoke Detection")

        # Timer
        self.timer = QTimer(self)
        self.timer.setInterval(1)
        self.video = None

        # Get the more robust path
        self.script_dir = str(Path(__file__).resolve().parent) 
        self.yolov5_repo_path = self.script_dir
        self.weights_dir = "runs/train/yolov5+se/weights/best.pt"

        # absolute path
        self.absolute_weights_path = str(Path(self.yolov5_repo_path) / self.weights_dir)

        self.model = None  # Initialize the model to None
        self.model_loading_thread = None  # Initialize the model loading thread to None
        self.image_detection_thread = None  # Initialize the image detection thread to None
        self.video_detection_worker_thread = None  # Initialize the video detection worker thread to None
        self.video_detection_worker = None  # Initialize the video detection worker to None

        # 创建和启动模型加载线程
        self.model_loading_thread = ModelLoadingThread(self.yolov5_repo_path, self.absolute_weights_path)

        # Connect the signal and slot for model loading
        self.model_loading_thread.model_loaded.connect(self.on_model_loaded)
        self.model_loading_thread.model_load_error.connect(self.on_model_load_error)

        # Start the model loading thread
        self.model_loading_thread.start()

        # Initialize the instance of imagedetection thread
        self.image_detection_thread = None
        # Initialize the instance of video detection thread
        self.video_detection_thread = None
        # Initialize the instance of video detection worker
        self.video_detection_worker = None

        # Binding
        self.bind_buttons()  # Bind buttons to their respective functions

    # Source control function for video 
    def stop_video(self):
        self.timer.stop()

        if self.video is not None and self.video.isOpened():
            self.video.release()
        self.video = None
        
    def closeEvent(self, event):

        self.stop_video()

        # 停止视频检测worker thread 
        if self.video_detection_worker_thread is not None and self.video_detection_worker_thread.isRunning():
            self.video_detection_worker.stop()
            self.video_detection_worker_thread.quit()
            self.video_detection_worker_thread.wait()
            # 清除对象引用
            self.video_detection_worker = None
            self.video_detection_worker_thread = None

        if self.image_detection_thread is not None and self.image_detection_thread.isRunning():
            self.image_detection_thread.stop()
            self.image_detection_thread.wait()  # Wait for the thread to finish
            self.image_detection_thread = None
        
        if self.model_loading_thread is not None and self.model_loading_thread.isRunning():
            self.model_loading_thread.stop()
            self.model_loading_thread.wait()
            self.model_loading_thread = None

        event.accept()  # Accept the close event
    
    def open_image(self):
        # Placeholder for image detection logic
        self.stop_video()
        if self.model is None:
            return
        if self.image_detection_thread is not None and self.image_detection_thread.isRunning():
            return
        file_path_tuple = QFileDialog.getOpenFileName(self, dir="./data/fire_smoke_datasets/images/train",filter="*.png ;*.jpg; *.jpeg ;*.bmp") 
        if file_path_tuple[0]:
            image_path = file_path_tuple[0]
            # get the absolute path of the image
            absoulute_path = str(Path(image_path).resolve())
        if isinstance(self.input, QLabel):
            pixmap = QPixmap(absoulute_path)
            label_width = self.input.width()
            label_highet = self.input.height()
            scaled_pixmap = pixmap.scaled(label_width, label_highet, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.input.setPixmap(scaled_pixmap)
            self.output.clear()

        # Create a new thread for image detection
        self.image_detection_thread = ImageDetectionThread(self.model, absoulute_path)
        # 连接线程信号到MainWindow的槽
        # 当发射完成信号时，调用on_image_detected slot
        self.image_detection_thread.detection_finished.connect(self.on_image_detected)
        # 当发射失败信号时，调用on_detection_error slot
        self.image_detection_thread.detection_error.connect(self.on_detection_error)
        # 启动线程
        self.image_detection_thread.start()
    
    # slot function of model loading thread
    @Slot(bool)
    def on_model_loaded(self, success):
        if success: # 如果线程报告加载测试成功 (接收到 True)
            try:
                # --- 在主线程中同步重新加载模型 ---
                # 使用在 __init__ 中存储的路径重新加载模型
                # 这个加载过程会暂时阻塞 UI，但由于模型加载线程已经确认路径和权重有效，
                # 这一步通常会比在线程中加载快（如果模型已经缓存）
                self.model = torch.hub.load(
                    self.yolov5_repo_path, # 你的 YOLOv5 仓库路径 
                    'custom',           # 你的 hubconf.py 入口点
                    path=self.absolute_weights_path, # 你的模型权重路径 
                    source='local'      # 从本地加载
                )
                # 模型在主线程中加载完成后，启用检测按钮
                # 检查按钮对象是否存在并有 setEnabled 方法
                if hasattr(self.img_detect, 'setEnabled'):
                    self.img_detect.setEnabled(True)
                if hasattr(self.video_detect, 'setEnabled'):
                    self.video_detect.setEnabled(True)

                # 清除加载信息，显示初始提示
                if isinstance(self.input, QLabel):
                     self.input.clear()
                     self.input.setText("请选择图片或视频进行检测")
                if isinstance(self.output, QLabel):
                     self.output.clear()
                     self.output.setText("检测结果将显示在这里")

                # --- 在模型加载成功后，创建并启动视频检测工作者线程 ---
                # 创建 QThread 实例
                self.video_detection_worker_thread = QThread()
                # 创建工作者对象，传递加载好的模型
                self.video_detection_worker = VideoDetectionWorker(self.model) # 将在主线程中加载的模型传递给工作者
                # 将工作者对象移动到新创建的线程
                self.video_detection_worker.moveToThread(self.video_detection_worker_thread)

                # 连接信号和槽：主线程信号 -> 工作者槽
                self.process_video_frame_signal.connect(self.video_detection_worker.process_frame)
                # 连接信号和槽：工作者信号 -> 主线程槽
                self.video_detection_worker.detection_finished.connect(self.on_video_frame_detected)
                self.video_detection_worker.detection_error.connect(self.on_video_detection_error) # 确保这个槽存在

                # 启动工作者线程的事件循环
                self.video_detection_worker_thread.start()
            except Exception as e:
                 pass
 
        # isFinished() 检查 run 方法是否完成
        if self.model_loading_thread is not None and self.model_loading_thread.isFinished():
             # wait() 阻塞直到线程结束
             self.model_loading_thread.wait()
             # 清理对象引用
             self.model_loading_thread = None
    
    # slot function of error signal of model loading thread
    @Slot(str)
    def on_model_load_error(self, error_message):
        if isinstance(self.input, QLabel):
            self.input.setText(f"Error: {error_message}") 
        if isinstance(self.output, QLabel):
            self.output.setText(f"Error: {error_message}")
        if self.model_loading_thread and self.model_loading_thread.isFinished():
            self.model_loading_thread.quit()
            self.model_loading_thread.wait()

    # slot function of img detection thread
    @Slot(np.ndarray)
    def on_image_detected(self, annotated_image_np):
        # receive the finished signal and the result image of detection thread
        qimg = convert2img(annotated_image_np)
        if isinstance(self.output, QLabel) and not qimg.isNull():
            pixmap = QPixmap.fromImage(qimg)
            label_width = self.output.width()
            label_highet = self.output.height()
            scaled_pixmap = QPixmap.fromImage(qimg).scaled(label_width, label_highet, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.output.setPixmap(scaled_pixmap)
        # when the detection is finished, stop the thread and release the resources
        if self.image_detection_thread and self.image_detection_thread.isFinished():
            self.image_detection_thread.wait()
            self.image_detection_thread = None
    
    @Slot(np.ndarray)
    def on_video_frame_detected(self, annotated_frame_np):
        # receive the finished signal and the result image of detection thread
        if annotated_frame_np is None or not isinstance(annotated_frame_np, np.ndarray):
            if isinstance(self.output, QLabel) and not qimg.isNull():
                self.output.clear()
            return
        # 将接收到的np数组转换成Qpixmap
        try:
            # 假设worker发送的是RGB np数组
            # 这里需要根据实际情况进行转换
            annotated_frame_rgb_for_display = cv2.cvtColor(annotated_frame_np, cv2.COLOR_BGR2RGB)
            qimg = convert2img(annotated_frame_rgb_for_display)
            if isinstance(self.output, QLabel) and not qimg.isNull():
                pixmap = QPixmap.fromImage(qimg)
                label_width = self.output.width()
                label_highet = self.output.height()
                scaled_pixmap = QPixmap.fromImage(qimg).scaled(label_width, label_highet, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.output.setPixmap(scaled_pixmap)
            else:
                if isinstance(self.output, QLabel):
                    self.output.clear()
        except Exception as e:
            if isinstance(self.output, QLabel):
                self.output.clear()
    
    # 接收错误信号
    @Slot(str)
    def on_video_detection_error(self, error_message):
        if isinstance(self.output, QLabel):
            self.output.setText(f"Error: {error_message}") 
        self.stop_video()

    @Slot(str)
    def on_detection_error(self, error_message):
        if isinstance(self.output, QLabel):
            self.output.setText(f"Error: {error_message}") 
        if self.image_detection_thread and self.image_detection_thread.isFinished():
            self.image_detection_thread.wait()
            self.image_detection_thread = None
            
    def vedio_detect(self):
        # Placeholder for image detection logic
        if not self.video or not self.video.isOpened():
            self.stop_video()
            return
        ret, frame_bgr = self.video.read()
        if not ret:
            self.stop_video()
            return
        
        # 主线程中显示原始帧
        frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.input.setPixmap(QPixmap.fromImage(convert2img(frame_rgb_display)))

        # 将原始BGR帧发送给worker thread进行推理
        if self.video_detection_worker_thread is not None and self.video_detection_worker_thread.isRunning() and self.video_detection_worker is not None:
            # 发送信号，将BGR格式的np数组发送给slot of worker thread
            self.process_video_frame_signal.emit(frame_bgr)
        else:
            self.stop_video()

    def open_video(self):
        # Placeholder for video detection logic
        self.stop_video()
        file_path = QFileDialog.getOpenFileName(self, dir="./data",filter="*.mp4; *.avi; *.mov")
        if file_path[0]:
            vedio_path = file_path[0]
            self.video = cv2.VideoCapture(vedio_path)
            if not self.video.isOpened():
                print("Error opening video file:{vedio_path}")
                self.video = None
                return
            
            fps = self.video.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                interval = max(1, int(1000 / fps))
                self.timer.setInterval(interval)
            else:
                default_interval = 33
                self.timer.setInterval(default_interval)

            self.timer.start()
        
    def bind_buttons(self):
        self.img_detect.clicked.connect(self.open_image)
        self.video_detect.clicked.connect(self.open_video) 
        self.timer.timeout.connect(self.vedio_detect)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 
