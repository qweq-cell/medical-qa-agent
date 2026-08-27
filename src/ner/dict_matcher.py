"""
基于词典的医疗实体识别（NER）
使用 medical_entity_dict.json 做前向最大匹配
"""
import json
import re
import os


class DictMatcher:
    """基于词典的最大前向匹配实体识别"""

    def __init__(self, dict_path: str, extra_entities: dict[str, list[str]] = None):
        """
        Args:
            dict_path: medical_entity_dict.json 的路径
            extra_entities: 额外实体映射，如 {"药品": ["二甲双胍", ...]}
        """
        self.entity_dict: dict[str, list[str]] = {}  # category -> [entities]
        self.trie: dict = {}  # 前缀树
        self._load(dict_path, extra_entities)

    def _load(self, dict_path: str, extra_entities: dict[str, list[str]] = None):
        """加载实体词典并构建前缀树

        Args:
            dict_path: 词典 JSON 路径
            extra_entities: 额外实体（不修改原词典文件）
        """
        if not os.path.exists(dict_path):
            print(f"[NER] 词典文件不存在: {dict_path}，使用空词典")
            return

        with open(dict_path, "r", encoding="utf-8") as f:
            self.entity_dict = json.load(f)

        # 合并额外实体
        if extra_entities:
            for cat, entities in extra_entities.items():
                if cat not in self.entity_dict:
                    self.entity_dict[cat] = []
                for ent in entities:
                    if ent not in self.entity_dict[cat]:
                        self.entity_dict[cat].append(ent)

        # 构建前缀树（用于快速匹配）
        # 同一个实体可能出现在多个类别中（如"2型糖尿病"既是疾病也是并发症）
        for category, entities in self.entity_dict.items():
            for entity in entities:
                if not entity:
                    continue
                node = self.trie
                for char in entity:
                    if char not in node:
                        node[char] = {}
                    node = node[char]
                # 存储所有类别，用列表支持多个类别
                if "$" not in node:
                    node["$"] = []
                if category not in node["$"]:
                    node["$"].append(category)

        # 类别优先级（当同一个实体出现在多个类别时，取优先级最高的）
        self._category_priority = {
            "疾病": 0, "症状": 1, "药品": 2, "检查": 3,
            "科室": 4, "并发症": 5, "食物": 6, "治疗方法": 7,
        }

    def extract(self, text: str, max_word_len: int = 20) -> list[dict]:
        """
        前向最大匹配提取实体

        Returns:
            [{"name": str, "category": str, "start": int, "end": int}, ...]
        """
        if not self.trie:
            return []

        results = []
        i = 0
        text_len = len(text)

        while i < text_len:
            # 跳过非中文字符开头的位置（优化性能）
            matched = None
            matched_len = 0
            node = self.trie

            # 前向匹配
            for j in range(i, min(i + max_word_len, text_len)):
                char = text[j]
                if char in node:
                    node = node[char]
                    if "$" in node:
                        # 取优先级最高的类别
                        categories = node["$"]
                        if len(categories) == 1:
                            matched = categories[0]
                        else:
                            matched = min(categories, key=lambda c: self._category_priority.get(c, 99))
                        matched_len = j - i + 1
                else:
                    break

            if matched:
                entity_name = text[i:i + matched_len]
                # 避免重复添加同一位置的实体
                if not results or results[-1]["start"] != i:
                    results.append({
                        "name": entity_name,
                        "category": matched,
                        "start": i,
                        "end": i + matched_len,
                    })
                i += matched_len
            else:
                i += 1

        # 后处理：合并重叠实体（长实体优先）
        results = self._merge_overlaps(results)

        return results

    def _merge_overlaps(self, entities: list[dict]) -> list[dict]:
        """合并重叠实体，长实体优先"""
        if not entities:
            return []

        # 按长度降序排列
        sorted_ents = sorted(entities, key=lambda e: e["end"] - e["start"], reverse=True)
        merged = []
        covered = set()

        for ent in sorted_ents:
            # 检查是否被已选实体覆盖
            pos_set = set(range(ent["start"], ent["end"]))
            if pos_set & covered:
                continue
            merged.append(ent)
            covered.update(pos_set)

        # 恢复原文顺序
        return sorted(merged, key=lambda e: e["start"])

    def get_entity_list(self, category: str) -> list[str]:
        """获取指定类型的实体列表"""
        return self.entity_dict.get(category, [])