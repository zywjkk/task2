import cv2
import numpy as np
import math

class shibie:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = None
        self.hsv_img = None
        self.processed_img = None
        self.combined_mask = None 
        self.color_ranges = {
            'Red': [
                ([0, 100, 100], [10, 255, 255]),     
                ([160, 100, 100], [179, 255, 255]),  
                ([165, 50, 150], [179, 150, 255])    
            ],
            'Green': [
                ([35, 60, 60], [85, 255, 255])
            ],
            'Blue': [
                ([100, 150, 100], [130, 255, 255])
            ]
        }
        
        self.draw_colors = {
            'Red': (0, 0, 255),
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0)
        }

    def load_image(self):
        """预处理：加载图像并转换色彩空间"""
        self.img = cv2.imread(self.image_path)
        if self.img is None:
            print(f"错误: 无法读取图像 '{self.image_path}'。")
            return False
            
        h, w = self.img.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            self.img = cv2.resize(self.img, (int(w * scale), int(h * scale)))
            
        self.bilateral = cv2.bilateralFilter(self.img, 9, 75, 75)
        self.hsv_img = cv2.cvtColor(self.bilateral, cv2.COLOR_BGR2HSV)
        self.combined_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        return True

    def get_refined_mask(self, color_name):
        """核心分割：内部灌浆 + 安全比例分水岭"""
        color_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        
        for lower, upper in self.color_ranges[color_name]:
            curr_mask = cv2.inRange(self.hsv_img, np.array(lower), np.array(upper))
            color_mask = cv2.bitwise_or(color_mask, curr_mask)
            
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        
        # 内部灌浆：寻找轮廓并填充满内部，彻底消除反光破洞
        cnts, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(color_mask)
        for c in cnts:
            if cv2.contourArea(c) > 100:  
                cv2.drawContours(solid_mask, [c], -1, 255, -1) 
        
        self.combined_mask = cv2.bitwise_or(self.combined_mask, solid_mask)
        
        # 动态深度提取种子
        sure_fg = np.zeros_like(solid_mask)
        dist = cv2.distanceTransform(solid_mask, cv2.DIST_L2, 5)
        
        cnts_solid, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts_solid:
            blob_mask = np.zeros_like(solid_mask)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            
            dist_blob = cv2.bitwise_and(dist, dist, mask=blob_mask)
            max_val = dist_blob.max()
            
            # 使用 13 保底，0.4 比例，切割
            thresh_val = max(13, 0.4 * max_val)
            _, fg_blob = cv2.threshold(dist_blob, thresh_val, 255, cv2.THRESH_BINARY)
            sure_fg = cv2.bitwise_or(sure_fg, np.uint8(fg_blob))
            
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        
        # 分水岭精细切割
        sure_bg = cv2.dilate(solid_mask, np.ones((3, 3), np.uint8))
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        clean_img = cv2.bitwise_and(self.img, self.img, mask=sure_bg)
        markers = cv2.watershed(clean_img, markers)
        
        final_cnts = []
        for marker_id in range(2, num_labels + 1):
            obj_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            obj_mask[markers == marker_id] = 255
            
            obj_mask = cv2.dilate(obj_mask, np.ones((3, 3), np.uint8))
            
            obj_cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if obj_cnts:
                largest_c = max(obj_cnts, key=cv2.contourArea)
                if cv2.contourArea(largest_c) > 400:
                    final_cnts.append(largest_c)
                    
        return final_cnts

    def classify_shape(self, contour, v_channel):
        """核心重构：3D光照判别法 + 花纹剥离"""
        area = cv2.contourArea(contour)
        if area < 400: return None
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: return None
        
        hull_peri = cv2.arcLength(hull, True)
        if hull_peri == 0: return None
        
        # 1. 提取基础几何特征
        (ccx, ccy), radius = cv2.minEnclosingCircle(hull)
        circle_area = math.pi * radius * radius
        circle_fit = hull_area / circle_area if circle_area > 0 else 0
        
        rect = cv2.minAreaRect(hull)
        (cx, cy), (w, h), angle = rect
        if w == 0 or h == 0: return None
        
        aspect_ratio = max(w, h) / min(w, h)
        rect_area = w * h
        extent = hull_area / rect_area 
        
        # 2. 3D 光照表面亮度分析
        obj_mask = np.zeros(v_channel.shape, dtype=np.uint8)
        cv2.drawContours(obj_mask, [contour], -1, 255, -1)
        
        dist = cv2.distanceTransform(obj_mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        
        is_curved_3d = False
        if max_dist >= 2:
            _, core_mask = cv2.threshold(dist, max_dist * 0.5, 255, cv2.THRESH_BINARY)
            core_mask = np.uint8(core_mask)
            
            _, inner_mask = cv2.threshold(dist, 2, 255, cv2.THRESH_BINARY)
            inner_mask = np.uint8(inner_mask)
            edge_mask = cv2.subtract(inner_mask, core_mask)
            
            if cv2.countNonZero(core_mask) > 0 and cv2.countNonZero(edge_mask) > 0:
                mean_core = cv2.mean(v_channel, mask=core_mask)[0]
                mean_edge = cv2.mean(v_channel, mask=edge_mask)[0]
                _, std_all = cv2.meanStdDev(v_channel, mask=inner_mask)
                stddev = std_all[0][0]
                
                brightness_diff = mean_core - mean_edge
                # 如果中心比边缘亮，或整体明暗方差大，说明是圆柱或球
                is_curved_3d = (brightness_diff > 20) or (stddev > 40)

        if is_curved_3d:
            # 抛弃粘连的花纹
            trim_d = max_dist * 0.3
            _, body_mask = cv2.threshold(dist, trim_d, 255, cv2.THRESH_BINARY)
            body_cnts, _ = cv2.findContours(np.uint8(body_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if body_cnts:
                main_body = max(body_cnts, key=cv2.contourArea)
                # 计算纯净主体的外接矩形
                (bcx, bcy), (bw, bh), bangle = cv2.minAreaRect(main_body)
                
                # 得到的矩形(rect)将是一个完全不被花纹干扰的，精准贴合物块真实的四角边界
                rect = ((bcx, bcy), (bw + 2 * trim_d, bh + 2 * trim_d), bangle)
                
                # 更新修正后的长宽比
                (cx, cy), (w, h), angle = rect
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

        # 为保留原有精细贴合的形状生成平滑凸包
        smoothed_hull = cv2.approxPolyDP(hull, 0.005 * hull_peri, True)
        
        # 强制找到外接矩形的四个角
        box = cv2.boxPoints(rect)
        ideal_rect_contour = np.int32(box).reshape((-1, 1, 2))

        if aspect_ratio < 1.28 and circle_fit > 0.80:
            if is_curved_3d:
                return "Ball", "poly", smoothed_hull
            else:
                return "Disk", "poly", smoothed_hull

        if aspect_ratio < 1.1:
            return "Cube", "poly", smoothed_hull

        if is_curved_3d:
            # 判定为圆柱体
            return "Cylinder", "rect", ideal_rect_contour
        else:
            return "Cuboid", "poly", smoothed_hull

    def analyze(self):
        if not self.load_image(): return
        
        self.processed_img = self.img.copy()
        
        color_counts = {'Red': 0, 'Green': 0, 'Blue': 0}
        shape_counts = {'Cuboid': 0, 'Disk': 0, 'Ball': 0, 'Cube': 0, 'Cylinder': 0}

        for color_name in self.color_ranges.keys():
            cnts = self.get_refined_mask(color_name)
            
            for cnt in cnts:
                res = self.classify_shape(cnt, self.hsv_img[:, :, 2])
                if res is None: continue
                
                shape_name, draw_type, ideal_contour = res
                
                color_counts[color_name] += 1
                if shape_name in shape_counts: 
                    shape_counts[shape_name] += 1
                
                # 绘制轮廓
                cv2.drawContours(self.processed_img, [ideal_contour], -1, self.draw_colors[color_name], 3, cv2.LINE_AA)
                
                M = cv2.moments(ideal_contour)
                if M["m00"] != 0:
                    tx = int(M["m10"] / M["m00"])
                    ty = int(M["m01"] / M["m00"])
                    label = f"{color_name}-{shape_name}"
                    
                    # 绘制UI
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    padding = 6
                    bg_tl = (tx - 40 - padding, ty - text_h - padding)
                    bg_br = (tx - 40 + text_w + padding, ty + baseline + padding - 2)
                    
                    cv2.rectangle(self.processed_img, bg_tl, bg_br, self.draw_colors[color_name], -1)
                    cv2.rectangle(self.processed_img, bg_tl, bg_br, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(self.processed_img, label, (tx-40, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        print("\n" + "="*45)
        print("识别完成：")
        print(f"图片中，红色有{color_counts['Red']}个，蓝色有{color_counts['Blue']}个，绿色有{color_counts['Green']}个；")
        print(f"长方体有{shape_counts['Cuboid']}个，圆盘有{shape_counts['Disk']}个，小球有{shape_counts['Ball']}个，"
              f"正方体有{shape_counts['Cube']}个，圆柱有{shape_counts['Cylinder']}个。")
        print("="*45 + "\n")
        
        self.show_result()

    def show_result(self):
        window_name = 'Recognized & Annotated result'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        h, w = self.img.shape[:2]
        win_h = 700
        win_w = int(w * (win_h / h))
        cv2.resizeWindow(window_name, win_w, win_h)

        cv2.imshow(window_name, self.processed_img)
        
        print("按键盘任意键退出")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = shibie("识别图像.pdf")
    analyzer.analyze()