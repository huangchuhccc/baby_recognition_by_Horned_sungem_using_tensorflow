# 单角蜂鸟单模型
import cv2, numpy, sys
sys.path.append('../../../')
import hsapi as hs # 导入 hsapi 模块, 注意导入路径

scale = 0.007843/2 # 图像预处理参数
mean = 0 # 图像预处理参数

device_list = hs.EnumerateDevices() # 获取所有已连接的角蜂鸟
device = hs.Device(device_list[0]) # 获取Device实例

device.OpenDevice() # 打开角蜂鸟设备

with open('/home/pi/SungemSDK-GraphModels/graphs/graph_baby_6_26', mode='rb') as f:
    data = f.read()
graph = device.AllocateGraph(data, scale, mean) # 获取Graph实例

try:
    while True:
        # 使用自带摄像头作为输入
        image = graph.GetImage(True) # 用角蜂鸟设备图像作为神经网络输入
        size=image.shape
        cx=int(size[0]/2)
        cy=int(size[1]/2)
        dx=int(size[0]*0.3)
        dy=int(size[1]*0.3)
        output, _ = graph.GetResult() # 获取神经网络输出结果
        if output[0] >0.5:
            label='baby is covered'
        else:
            label='baby is not covered'
        cv2.putText(image,label,(cy-dy, cx-int(dx*1.1)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        print(output)
        cv2.rectangle(image,(cy-dy, cx-dx), (cy+dy, cx+dx),(255,255,0),5)
        cv2.imshow("baby_recgonition", image)
        cv2.waitKey(1)
finally:
    graph.DeallocateGraph() # 释放神经网络资源
    device.CloseDevice() # 关闭角蜂鸟设备