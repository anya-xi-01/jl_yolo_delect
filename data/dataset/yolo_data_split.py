import os
import shutil
import random

# 设置原始数据路径
images_dir = 'clahe/images'
labels_dir = 'clahe/labels'

# 设置划分后的输出路径
output_base = './clahe'
train_images_dir = os.path.join(output_base, 'train', 'images')
train_labels_dir = os.path.join(output_base, 'train', 'labels')
val_images_dir = os.path.join(output_base, 'val', 'images')
val_labels_dir = os.path.join(output_base, 'val', 'labels')

# 创建输出目录
for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
    os.makedirs(directory, exist_ok=True)

# 获取所有图片文件
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 打乱顺序
random.shuffle(image_files)

# 计算划分数量
total = len(image_files)
train_count = int(total * 0.8)

# 划分
train_files = image_files[:train_count]
val_files = image_files[train_count:]

# 拷贝文件
def copy_files(file_list, target_images_dir, target_labels_dir):
    for image_file in file_list:
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + '.txt'

        # 拷贝图片
        src_img = os.path.join(images_dir, image_file)
        dst_img = os.path.join(target_images_dir, image_file)
        shutil.copy2(src_img, dst_img)

        # 拷贝对应的标签（如果存在）
        src_lbl = os.path.join(labels_dir, label_file)
        dst_lbl = os.path.join(target_labels_dir, label_file)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# 执行拷贝
copy_files(train_files, train_images_dir, train_labels_dir)
copy_files(val_files, val_images_dir, val_labels_dir)

print(f"数据集划分完成，共 {total} 张图片")
print(f"训练集：{len(train_files)}，验证集：{len(val_files)}")
