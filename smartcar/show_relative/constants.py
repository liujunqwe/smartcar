# 定义车辆类型对应的 y 方向真实距离（单位：米）
VEHICLE_TYPE_DISTANCE = {
    'Truck': 10.2,
    'SUV': 6.2,
    'Sedan': 5.8,
    'Microbus': 8.6,  # 根据实际情况设置
    'Minivan': 8.5,  # 根据实际情况设置
    'Bus': 9.8
}

# 默认距离，如果检测到的类别不在上面定义的字典中
DEFAULT_DISTANCE = 2.0

def y_to_depth(y, y_min, y_max, depth_min, depth_max):
    """
    将y坐标映射到真实世界的深度（米）。
    :param y: 当前车辆的y坐标（像素）
    :param y_min: 视频中最上方的y坐标（像素）
    :param y_max: 视频中最下方的y坐标（像素）
    :param depth_min: y_max对应的最小深度（离摄像头最近，米）
    :param depth_max: y_min对应的最大深度（离摄像头最远，米）
    :return: 车辆的真实深度（米）
    """
    if y_max == y_min:
        return depth_min  # 避免除以零
    # 线性映射
    depth = depth_max - (y - y_min) * (depth_max - depth_min) / (y_max - y_min)
    return depth 