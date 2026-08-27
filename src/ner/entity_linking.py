"""
实体链接 — 将识别到的实体标准化
"""
import json
import os


class EntityLinker:
    """实体标准化链接"""

    def __init__(self, dict_path: str):
        """
        Args:
            dict_path: medical_entity_dict.json 路径
        """
        self.entity_dict: dict[str, list[str]] = {}
        self._load(dict_path)

        # 别名映射（常见变体 → 标准名）
        self.alias_map: dict[str, str] = {
            # 疾病别名
            "2型糖尿病": "II型糖尿病",
            "糖尿病": "糖尿病",
            "糖病": "糖尿病",
            "高血糖": "糖尿病",
            "血压高": "高血压",
            "血压升高": "高血压",
            "感冒": "上呼吸道感染",
            "肺结核": "结核病",
            "乙肝": "乙型病毒性肝炎",
            "甲肝": "甲型病毒性肝炎",
            "丙肝": "丙型病毒性肝炎",
            "冠心病": "冠状动脉粥样硬化性心脏病",
            "心梗": "急性心肌梗死",
            "脑梗": "脑梗死",
            "中风": "脑卒中",
            "高血压病": "高血压",
            # 症状别名
            "发烧": "发热",
            "咳嗽": "咳嗽",
            "拉肚子": "腹泻",
            "便秘": "便秘",
            "头疼": "头痛",
            "头晕": "眩晕",
            "肚子疼": "腹痛",
            "胃疼": "胃痛",
            "口渴多饮": "口渴多饮",
            "多饮": "多饮",
            "多尿": "多尿",
            # 检查别名
            "空腹血糖": "血糖",
            "血糖": "血糖",
            "血常规": "血常规",
            "尿常规": "尿常规",
            # 药品别名
            "二甲双胍": "盐酸二甲双胍片",
            "阿司匹林": "阿司匹林",
            "氨氯地平": "氨氯地平",
            "硝苯地平": "硝苯地平",
            "卡托普利": "卡托普利",
            "胰岛素": "胰岛素",
        }

    def _load(self, dict_path: str):
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                self.entity_dict = json.load(f)

    def link(self, name: str, category: str) -> str:
        """
        将实体名标准化为标准名称

        Args:
            name: 原始实体名
            category: 实体类型

        Returns:
            标准化后的名称
        """
        # 1. 检查别名映射
        if name in self.alias_map:
            return self.alias_map[name]

        # 2. 检查是否已在标准词典中
        std_list = self.entity_dict.get(category, [])
        if name in std_list:
            return name

        # 3. 模糊匹配：找包含关系
        for std_name in std_list:
            if name in std_name or std_name in name:
                return std_name

        # 4. 找不到则返回原名称
        return name