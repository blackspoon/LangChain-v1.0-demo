from dashscope import ImageSynthesis
import base64

# 1. 调用通义·万相模型（Wanx-v1）
result = ImageSynthesis.call(
    model="wanx-v1",    # 通义·万相模型
    prompt="在阳光下的现代城市街头，一只戴着墨镜的橙色猫咪喝咖啡，插画风格。",
    api_key="xxxxxxxx",
    size="1024*1024",   # 图片大小
)

# 2. 输出结果
if result.status_code == 200:
    # 有的版本返回 Base64，有的返回 URL，这里兼容两种
    output = result.output
    if "b64_json" in output["results"][0]:
        image_base64 = output["results"][0]["b64_json"]
        image_data = base64.b64decode(image_base64)
        with open("qwen_image.png", "wb") as f:
            f.write(image_data)
        print("✅ 已生成图片：qwen_image.png")
    elif "url" in output["results"][0]:
        print("✅ 图片生成成功：", output["results"][0]["url"])
    else:
        print("⚠️ 未找到可解析的图片数据：", output)
else:
    print("❌ 调用失败：", result.message)
