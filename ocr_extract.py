"""
OCR 文字提取脚本 — 使用 PaddleOCR
处理甲骨文校释卷图片，在原文件处生成同名的 .txt 文件
"""
import os
import sys

# 必须在 import paddle 之前禁用 oneDNN
os.environ["FLAGS_use_onednn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from paddleocr import PaddleOCR

DATA_DIR = r"C:\Users\zgb\Desktop\graphrag\数据"

# 初始化 PaddleOCR
print("正在初始化 PaddleOCR ...")
ocr = PaddleOCR(lang="ch")
print("初始化完成！\n")

# 收集所有图片
images = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("ocr")
])

print(f"找到 {len(images)} 张图片:\n")
for img in images:
    print(f"  - {img}")

print("\n" + "=" * 60)

for img in images:
    img_path = os.path.join(DATA_DIR, img)
    txt_path = os.path.splitext(img_path)[0] + ".txt"

    print(f"\n处理: {img} ...")
    try:
        # 使用 predict 方法（新版 API）
        result = ocr.predict(img_path)

        lines = []
        # PaddleOCR 3.x predict 返回格式处理
        for page in result:
            rec_texts = []
            if hasattr(page, 'rec_texts'):
                rec_texts = page.rec_texts
            elif isinstance(page, dict):
                rec_texts = page.get('rec_texts', [])
            elif isinstance(page, list):
                # 可能是旧格式 [[[bbox], (text, confidence)], ...]
                for item in page:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        if isinstance(item[1], (list, tuple)):
                            text = item[1][0] if item[1] else ""
                            confidence = item[1][1] if len(item[1]) > 1 else 0
                        else:
                            text = str(item[1])
                            confidence = 0
                    else:
                        text = str(item)
                        confidence = 0
                    rec_texts.append(text)

            for text in rec_texts:
                text = str(text)
                print(f"  {text}")
                lines.append(text)

        full_text = "\n".join(lines)

        # 写入 txt
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"  ✅ 已保存: {os.path.basename(txt_path)} ({len(lines)} 行)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ❌ 错误: {e}", file=sys.stderr)

print("\n" + "=" * 60)
print("全部完成！")
