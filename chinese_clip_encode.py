from translate import Translator


class ChineseCLIPEncode:
    """
    用于 CLIP 的中英文编码节点。

    该节点接收一个 CLIP 模型作为输入，并包含一个文本区域，用于输入中文或英文文本。
    如果输入的是中文，将使用 `translate` 库将其翻译成英文，并将翻译结果打印到控制台。
    此节点使用 ComfyUI 的 CLIP 文本编码方式对文本进行编码，输出 CONDITIONING 类型数据，以便于与 KSampler 等节点连接。
    """

    @classmethod
    def INPUT_TYPES(s):
        # 定义节点的输入类型
        return {
            "required": {
                "clip": ("CLIP",),  # CLIP 模型
                "text": ("STRING", {"multiline": True, "default": ""}),  # 多行文本框，默认值为空字符串
            },
        }

    # 定义节点的输出类型
    RETURN_TYPES = ("CONDITIONING",)  # 输出 CONDITIONING 类型数据
    FUNCTION = "encode"  # 节点的入口函数为 "encode"

    CATEGORY = "AI_Boy"  # 节点所属类别为 "AI_Boy"

    def encode(self, clip, text):
        """
        对输入文本进行翻译然后进行 CLIP 编码。

        参数：
            clip: CLIP 模型。
            text (str): 待编码的文本。

        返回值：
            CONDITIONING: 编码后的 CONDITIONING 数据。
        """

        # 判断文本是否包含中文
        if self.is_chinese(text):
            # 如果包含中文，则将其翻译成英文。必须指定 from_lang 参数，否则翻译无效
            translator = Translator(to_lang="en", from_lang="zh")
            translated_text = translator.translate(text)  # 进行翻译
            print(f"翻译结果：{translated_text}")  # 打印翻译结果到控制台

            text = translated_text  # 将翻译后的文本赋给 text 变量，用于后续的 CLIP 编码

        # 进行 CLIP 文本编码
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        return ([[cond, {"pooled_output": pooled}]],)

    @staticmethod
    def is_chinese(text):
        """
        检查输入文本是否包含中文字符。

        参数：
            text (str): 待检查的文本。

        返回值：
            bool: 如果文本包含中文字符，则返回 True，否则返回 False。
        """
        # 遍历文本中的每个字符
        for char in text:
            # 判断字符的 Unicode 编码是否在中文范围内
            if '\u4e00' <= char <= '\u9fff':
                return True  # 如果包含中文字符，则返回 True
        return False  # 如果没有找到中文字符，则返回 False


# 包含要导出的所有节点及其名称的字典
# 注意：名称应全局唯一
NODE_CLASS_MAPPINGS = {
    "ChineseCLIPEncode": ChineseCLIPEncode,  # 将 ChineseCLIPEncode 类注册为名为 "ChineseCLIPEncode" 的节点
}

# 节点名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChineseCLIPEncode": "ChineseCLIPEncode",
}
