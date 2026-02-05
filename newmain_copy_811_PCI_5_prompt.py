import os
import json
import requests
import base64
import time
import requests.exceptions
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ==================== 1. 外置规则配置（独立维护，符合JTG 5210-2018）====================
DISEASE_RULES = {
    "龟裂": {
        "轻度": {"weight": 0.6, "desc": "缝宽 <2mm"},
        "中度": {"weight": 0.8, "desc": "缝宽2-5mm"},
        "重度": {"weight": 1.0, "desc": "缝宽 >5mm"}
    },
    "横向裂纹": {
        "轻度": {"weight": 0.6, "width": 0.2, "desc": "缝宽≤3mm"},
        "重度": {"weight": 1.0, "width": 0.2, "desc": "缝宽 >3mm"}
    },
    "纵向裂纹": {
        "轻度": {"weight": 0.6, "width": 0.2, "desc": "缝宽≤3mm"},
        "重度": {"weight": 1.0, "width": 0.2, "desc": "缝宽 >3mm"}
    },
    "坑槽": {
        "轻度": {"weight": 0.8, "desc": "面积 <0.1m²"},
        "重度": {"weight": 1.0, "desc": "面积≥0.1m²"}
    },
    "麻面": {
        "轻度": {"weight": 0.6, "desc": "颗粒轻微脱落"},
        "重度": {"weight": 1.0, "desc": "颗粒严重脱落"}
    },
    "修补": {
        "一般": {"weight": 0.1, "desc": "修补区域"}  
    },
    "积水": {
        "轻度": {"weight": 0.6, "desc": "深度10-25mm"},
        "重度": {"weight": 1.0, "desc": "深度 >25mm"}
    }
}

GRADE_THRESHOLDS = [
    (90, "优"),
    (80, "良"),
    (70, "中"),
    (60, "次"),
    (0, "差")
]


SIMPLE_PROMPT = """
角色：道路病害检测专家
依据《JTG 5210—2018 公路技术状况评定标准》分析图像。
【核心要求】
1. 优先读取图像中已有的检测框及其标注数据（病害类型、面积等），此类病害直接采用框内数值，不重新估算
2. 同时独立识别全图，若发现检测框未覆盖的其他病害，需补充识别并基于车道线宽度(3.5m)估算其物理尺寸
3. 识别7类病害：龟裂、横向裂纹、纵向裂纹、坑槽、麻面、修补、积水
对每类病害输出：
type: 病害中文名称（严格使用上述7类名称）
severity: 程度（龟裂/坑槽/麻面/积水用"轻度/中度/重度"；修补用"一般"；裂纹用"轻度/重度"）
area: 面积(m²)，保留1位小数（检测框内病害直接采用框内标注值；新增病害需估算）
length: 仅裂纹类必填（横向/纵向裂纹），单位m；非裂纹类设为null
估算图像中可见路面的总面积 total_area_A (m²)，保留1位小数
description: 路面状况简要描述(<30字)，需说明是否包含框外新增病害
【输出格式】严格JSON，仅包含以下字段：
{
 "description": "字符串",
 "total_area_A": 45.0,
 "findings": [
   {"type": "龟裂", "severity": "重度", "area": 2.5, "length": null},
   {"type": "横向裂纹", "severity": "轻度", "area": 0.8, "length": 4.0}
 ]
}
严禁包含任何额外文本、Markdown或解释。
"""


def calculate_metrics(findings: List[Dict], total_area: float) -> Tuple[float, float, str]:
    """
    根据识别结果精确计算DR/PCI/等级
    :return: (DR, PCI, grade)
    """
    if total_area <= 0:
        return 0.0, 100.0, "优"
    
    weighted_sum = 0.0

    for item in findings:
        disease = item["type"]
        severity = item["severity"]
        area = float(item.get("area", 0))
        length = item.get("length")
        
        # 裂纹类面积校验：若模型未计算area，用length*0.2补全
        if disease in ["横向裂纹", "纵向裂纹"] and (area <= 0 or area is None):
            if length and float(length) > 0:
                area = float(length) * 0.2
                item["area"] = round(area, 1)  # 回写修正值
        
        # 获取权重
        try:
            if disease == "修补":
                weight = DISEASE_RULES[disease]["一般"]["weight"]
            else:
                weight = DISEASE_RULES[disease][severity]["weight"]
        except KeyError:
            weight = 1.0  # 未知类型默认重度
        
        weighted_sum += weight * area

    # DR计算
    dr = 100 * (weighted_sum / total_area)
    dr = min(max(dr, 0.0), 100.0)  # 边界保护

    # PCI计算（JTG 5210-2018公式）
    pci = 100 - 15.00 * (dr ** 0.412)
    pci = min(max(pci, 0.0), 100.0)
    
    # 等级判定
    grade = next((g for t, g in GRADE_THRESHOLDS if pci >= t), "差")

    return round(dr, 2), round(pci, 2), grade


url = "http://10.10.99.235:31013/model/api/openai/inspect-qwen2.5-vl-32b-instruct/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 10001"
}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_mllm_for_detection(encoded_image: str, max_retries: int = 3) -> Dict:
    """
    调用MLLM进行病害识别，支持网络错误自动重试
    :param encoded_image: Base64编码的图像
    :param max_retries: 最大重试次数（默认3次）
    :return: 识别结果或错误信息
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SIMPLE_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                        ]
                    }
                ],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 1024
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # 鲁棒性清洗：提取JSON
            if "{" in content and "}" in content:
                content = content[content.find("{"):content.rfind("}") + 1]
            
            raw_result = json.loads(content)
            
            # 验证必要字段
            required = ["description", "total_area_A", "findings"]
            if not all(k in raw_result for k in required):
                raise ValueError(f"缺失必要字段，实际返回: {list(raw_result.keys())}")
            
            return raw_result  # 成功返回结果
            
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.RequestException,
                ConnectionResetError) as e:
            last_error = e
            wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s
            
            if attempt < max_retries - 1:
                print(f"  ⚠️  网络错误（尝试 {attempt + 1}/{max_retries}）: {str(e)[:80]}，{wait_time}秒后重试...")
                time.sleep(wait_time)
            continue  # 继续下一次重试
            
        except Exception as e:
            # 非网络错误（如JSON解析失败）直接返回，不重试
            return {
                "error": "MLLM识别失败",
                "message": str(e),
                "raw_response": content if 'content' in locals() else None,
                "attempt": attempt + 1
            }
    
    # 所有重试均失败
    return {
        "error": "Processing error",
        "details": str(last_error),
        "retries": max_retries,
        "status": "failed_after_retries"
    }

def process_single_image(image_path: str) -> Dict:
    """处理单张图片：识别 + 精确计算"""
    try:
        # Step 1: 模型识别（自动重试）
        encoded = encode_image_to_base64(image_path)
        detection_result = call_mllm_for_detection(encoded)
        
        if "error" in detection_result:
            return detection_result
        
        # Step 2: 后处理计算
        findings = detection_result.get("findings", [])
        total_area = float(detection_result.get("total_area_A", 0))
        
        dr, pci, grade = calculate_metrics(findings, 100)
        
        # Step 3: 组装最终结果
        return {
            "description": detection_result["description"],
            "findings": findings,
            "total_area_A": round(total_area, 1),
            "DR": dr,
            "PCI": pci,
            "grade": grade
        }
        
    except Exception as e:
        return {
            "error": "处理异常",
            "message": str(e),
            "traceback": __import__("traceback").format_exc()
        }

def main():
    # 配置路径
    folders = [r"D:\kp\海康数据\normal_0feac152-f8fd-4c35-9ae1-362cceef16b1_zip"]
    output_json_path = r"D:\kp\海康数据\mllm_results_new_PCI-5_prom_all.json"
    results = {}

    for folder in folders:
        if not os.path.isdir(folder):
            print(f" 跳过不存在的文件夹: {folder}")
            continue
            
        print(f"\n扫描目录: {folder}")
        image_files = [
            f for f in os.listdir(folder) 
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ]
        
        print(f"共发现 {len(image_files)} 张图片")
        for idx, file_name in enumerate(image_files, 1):
            file_path = os.path.join(folder, file_name)
            print(f"[{idx}/{len(image_files)}] 处理: {file_name}")
            
            result = process_single_image(file_path)
            results[file_name] = result
            
            # 实时保存（防中断丢失）
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 最终保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n 处理完成！结果已保存至:\n{output_json_path}")

    # 统计摘要
    valid_results = [r for r in results.values() if "error" not in r]
    if valid_results:
        avg_pci = sum(r["PCI"] for r in valid_results) / len(valid_results)
        grade_counts = {}
        for r in valid_results:
            grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1
        
        print(f"\n 统计摘要:")
        print(f"   有效图片: {len(valid_results)}/{len(results)}")
        print(f"   平均PCI: {avg_pci:.2f}")
        print(f"   等级分布: {dict(sorted(grade_counts.items()))}")

if __name__ == "__main__":
    main()