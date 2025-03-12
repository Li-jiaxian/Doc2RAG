import warnings
from typing import Any, Iterator, List, Mapping, Optional

import numpy as np
import pdfplumber.page
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document
from PIL import Image

from rag.module.indexing.loader.utils.ocr import extract_from_images_with_rapidocr

_PDF_FILTER_WITH_LOSS = ["DCTDecode", "DCT", "JPXDecode"]
_PDF_FILTER_WITHOUT_LOSS = [
    "LZWDecode",
    "LZW",
    "FlateDecode",
    "Fl",
    "ASCII85Decode",
    "A85",
    "ASCIIHexDecode",
    "AHx",
    "RunLengthDecode",
    "RL",
    "CCITTFaxDecode",
    "CCF",
    "JBIG2Decode",
]


class CustomizedPDFPlumberLoader(PDFPlumberLoader):
    """使用 PDFPlumber 加载 PDF 文件的自定义加载器。

    支持单列和双列PDF文档的智能识别和处理。

    Attributes:
        paged (bool): 是否对PDF文件进行分页处理。默认为False。
    """

    paged: bool = False

    def load(self) -> List[Document]:
        """Load file."""
        parser = PDFPlumberParser(
            text_kwargs=self.text_kwargs,
            dedupe=self.dedupe,
            extract_images=self.extract_images,
            paged=self.paged,
        )
        blob = Blob.from_path(self.file_path)  # type: ignore[attr-defined]
        return parser.parse(blob)


class PDFPlumberParser(BaseBlobParser):
    """使用 PDFPlumber 解析 PDF 文件的解析器。

    支持单双列布局自动识别、图片提取和OCR功能。

    Attributes:
        PDF_OCR_THRESHOLD (tuple): OCR处理的图片尺寸阈值, (宽度比例, 高度比例)
        text_kwargs (dict): 传递给pdfplumber.Page.extract_text()的关键字参数
        dedupe (bool): 是否去除重复字符
        extract_images (bool): 是否提取图片并进行OCR
        paged (bool): 是否按页返回文档
    """

    PDF_OCR_THRESHOLD = (0.4, 0.4)

    def __init__(
        self,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        dedupe: bool = False,
        extract_images: bool = False,
        paged: bool = False,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images
        self.paged = paged

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:  # type: ignore[valid-type]
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:  # type: ignore[attr-defined]
            # 打开PDF文档
            doc = pdfplumber.open(file_path)

            # 从第 i 页开始遍历
            i = 1
            for page in doc.pages[i - 1 :]:
                # 获取页面的宽度和高度
                width = page.width
                # 通过分析文本框的数量来判断栏数
                text_boxes = page.extract_words()
                num_columns = self.determine_columns(text_boxes, width)
                if num_columns == 2:  # 如果是双栏, 那就走我写的功能, 单栏就走
                    content = self._process_double_column_page(page)
                    # content = self._process_page_content(page)
                    print(f"第{i}页: 双")
                    # 提取页面内容和图片内容
                else:
                    content = self._process_page_content(page)
                    print(f"第{i}页: 单")
                i += 1
                image_content = self._extract_images_from_page(page)

                # 构建元数据
                metadata = {
                    "source": blob.source,  # type: ignore[attr-defined]
                    "file_path": blob.source,  # type: ignore[attr-defined]
                    "page": page.page_number - 1,
                    "total_pages": len(doc.pages),
                }

                # 添加文档原始元数据中的字符串和整数类型数据
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int)):
                        metadata[key] = value

                # 生成文档对象
                yield Document(page_content=content + "\n\n" + image_content, metadata=metadata)

    def parse(self, blob: Blob) -> List[Document]:
        docs = list(self.lazy_parse(blob))
        if self.paged:
            return docs
        else:
            metadata = {
                "source": blob.source,  # type: ignore[attr-defined]
                "file_path": blob.source,  # type: ignore[attr-defined]
                "total_pages": len(docs),
            }
            page_content = "\n\n".join([doc.page_content for doc in docs])
            return [Document(page_content=page_content, metadata=metadata)]

    def determine_columns(self, text_boxes, width) -> int:
        """判断PDF页面是单列还是双列布局。

        该方法通过分析页面中文本框的分布来判断PDF页面的布局是单列还是双列。
        算法步骤如下：
        1. 如果页面没有文本框，默认返回单列（1）。
        2. 计算每个文本框的x坐标中点，存储在x_centers列表中。
        3. 将页面宽度分成两半，计算中点位置mid_point。
        4. 初始化左右两侧文本框计数器left_count和right_count。
        5. 遍历x_centers，根据文本框中点与页面中点的距离，统计左右两侧的文本框数量。
           - 如果文本框中点小于(mid_point - threshold)，则计入左侧。
           - 如果文本框中点大于(mid_point + threshold)，则计入右侧。
           - threshold是一个阈值，用于避免靠近中线的文本框影响判断。
        6. 计算左右两侧文本框数量占总文本框数量的比例。
        7. 如果右侧文本框比例大于设定的阈值（0.4），则认为页面为双列（2）。
           否则，认为页面为单列（1）。

        Args:
            text_boxes (list): 页面中的文本框列表
            width (float): 页面宽度

        Returns:
            int: 1表示单列, 2表示双列
        """
        # 计算文本框的分布，判断栏数
        # 这里可以根据文本框的x坐标来判断
        if not text_boxes:
            return 1

        # 获取所有文本框的x坐标中点
        x_centers = [(box["x0"] + box["x1"]) / 2 for box in text_boxes]

        # 将页面宽度分成两半
        mid_point = width / 2
        left_count = 0
        right_count = 0

        # 统计左右两侧的文本框数量
        threshold = width * 0.2  # 设置一个阈值，避免靠近中线的文本影响判断
        for x in x_centers:
            if x < (mid_point - threshold):
                left_count += 1
            elif x > (mid_point + threshold):
                right_count += 1

        # 如果左右两侧都有足够数量的文本框，判定为双栏
        # 这里设置一个最小文本框数量的阈值，避免噪声影响
        threshold = 0.4
        left_proportions = left_count / len(x_centers)
        right_proportions = right_count / len(x_centers)
        print(f"left: {left_proportions}, right: {right_proportions}")
        # if threshold < right_proportions < (1 - threshold) : # 右边占据0.4到0.6, 也就是两边差不多, 那就认为是双栏
        if right_proportions > threshold:  # 右边占据0.4到0.6, 也就是两边差不多, 那就认为是双栏
            return 2
        return 1

    def _process_double_column_page(self, page: pdfplumber.page.Page) -> str:
        """处理双列页面内容。

        将页面分为左右两列, 分别提取文本后合并。

        Args:
            page: PDF页面对象

        Returns:
            str: 处理后的页面文本内容
        """
        words = page.extract_words()
        left_words = []
        right_words = []
        mid_point = page.width / 2
        for word in words:
            if word["x0"] < mid_point:
                left_words.append(word)
            else:
                right_words.append(word)
        # 单词合并成列
        left_lines = self._merge_words(left_words)
        right_lines = self._merge_words(right_words)
        # 列合并成
        return "\n".join(left_lines + ["\n\n"] + right_lines)

    def _merge_words(self, words) -> list:
        """将单词合并成行。

        基于垂直位置将属于同一行的单词组合在一起。

        Args:
            words: 单词列表, 每个单词包含位置信息

        Returns:
            list: 合并后的文本行列表
        """
        lines = []
        if not words:
            return lines

        # 按y坐标排序
        current_line = [words[0]]
        y_threshold = 3  # 允许的y坐标差异阈值

        for word in words[1:]:
            # 如果当前单词与当前行第一个单词的y坐标差异在阈值内
            if abs(word["top"] - current_line[0]["top"]) <= y_threshold:
                current_line.append(word)
            else:
                # 按x坐标排序当前行的单词并加入到结果中
                current_line.sort(key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in current_line)
                lines.append(line_text)
                current_line = [word]

        # 处理最后一行
        if current_line:
            current_line.sort(key=lambda w: w["x0"])
            line_text = " ".join(w["text"] for w in current_line)
            lines.append(line_text)

        return lines

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        return page.extract_text(**self.text_kwargs)

    def _extract_images_from_page(self, page: pdfplumber.page.Page) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""

        images = []
        for img in page.images:
            try:
                img_filter = img["stream"]["Filter"]
                if isinstance(img_filter, list):
                    filter_names = [f.name for f in img_filter]
                else:
                    filter_names = [img_filter.name]

                try:
                    img_data = img["stream"].get_data()
                    stream = img["stream"]
                    width = stream["Width"]
                    height = stream["Height"]
                    size = (width, height)
                    if (
                        width / page.width < self.PDF_OCR_THRESHOLD[0]
                        or height / page.height < self.PDF_OCR_THRESHOLD[1]
                    ):
                        continue
                    # 处理无损压缩的图片
                    if any(f in _PDF_FILTER_WITHOUT_LOSS for f in filter_names):
                        # 根据不同的色彩空间创建图片
                        color_space_map = {
                            "DeviceGray": "L",
                            "DeviceRGB": "RGB",
                            "DeviceCMYK": "CMYK",
                        }

                        if stream["BitsPerComponent"] == 1:
                            img_array = np.array(
                                Image.frombytes("1", size, img_data).convert("L")
                            )
                        elif stream["ColorSpace"] in color_space_map:
                            print(f"处理{stream['ColorSpace']}色彩空间的图片")
                            mode = color_space_map[stream["ColorSpace"]]
                            img_array = np.array(Image.frombytes(mode, size, img_data))
                        else:
                            # 处理其他未知格式
                            img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(
                                height, width, -1
                            )
                        images.append(img_array)

                    # 处理有损压缩的图片
                    elif any(f in _PDF_FILTER_WITH_LOSS for f in filter_names):
                        images.append(img_data)
                    else:
                        warnings.warn(f"未知的PDF过滤器类型: {filter_names}")
                except (ValueError, KeyError) as e:
                    warnings.warn(f"处理图片数据时出错: {str(e)}")
                    continue

            except Exception as e:
                warnings.warn(f"提取PDF页面图片时出错: {str(e)}")
                continue
        return extract_from_images_with_rapidocr(images)
