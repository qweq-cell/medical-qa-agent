"""
MedGuardian 配置
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
KG_JSON = os.path.join(DATA_DIR, "QASystemOnMedicalKG", "data", "medical.json")
ENTITY_DICT_FILE = os.path.join(PROCESSED_DIR, "medical_entity_dict.json")

# Neo4j 知识图谱（本地实例，Neo4j Desktop "Intelligent grad"）
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# LLM 配置（key 从项目根 .env 读取，不硬编码；.env 已被 .gitignore 忽略）
try:
    from src.llm.client import load_env_file
    load_env_file()
except Exception:
    pass
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_ENABLED = bool(LLM_API_KEY and LLM_API_KEY.startswith("sk-"))

# 性别相关关键词
MALE_KEYWORDS = ["男", "男性", "先生"]
FEMALE_KEYWORDS = ["女", "女性", "女士"]
FEMALE_ONLY_TESTS = ["子宫", "卵巢", "阴道", "宫颈", "输卵管", "乳腺", "妇科", "子宫肌瘤", "卵巢囊肿"]
MALE_ONLY_TESTS = ["前列腺", "睾丸", "阴茎", "精囊", "前列腺增生"]
FEMALE_ONLY_DRUGS = ["雌二醇", "孕酮", "黄体酮", "己烯雌酚", "戊酸雌二醇"]
MALE_ONLY_DRUGS = ["非那雄胺", "前列康", "甲睾酮", "度他雄胺"]

# 实体类型映射
ENTITY_CATEGORIES = ["疾病", "症状", "药品", "检查", "科室", "食物", "并发症", "治疗方法"]

# 额外实体（词典中缺失但常见的医疗实体）
EXTRA_ENTITIES = {
    "药品": [
        "二甲双胍", "阿司匹林", "氨氯地平", "硝苯地平", "卡托普利",
        "胰岛素", "布洛芬", "奥美拉唑", "阿莫西林", "头孢克肟",
        "头孢拉定", "头孢氨苄", "罗红霉素", "左氧氟沙星", "青霉素",
        "阿奇霉素", "红霉素", "克拉霉素", "甲硝唑", "替硝唑",
        "氟康唑", "伊曲康唑", "辛伐他汀", "阿托伐他汀", "瑞舒伐他汀",
        "氯吡格雷", "华法林", "利伐沙班", "达比加群", "普伐他汀",
        "非洛地平", "贝那普利", "缬沙坦", "厄贝沙坦", "替米沙坦",
        "美托洛尔", "比索洛尔", "普萘洛尔", "螺内酯", "氢氯噻嗪",
        "呋塞米", "地高辛", "硝酸甘油", "单硝酸异山梨酯",
        "阿卡波糖", "格列本脲", "格列美脲", "瑞格列奈", "吡格列酮",
        "西格列汀", "利拉鲁肽", "达格列净", "恩格列净",
        "他莫昔芬", "来曲唑", "阿那曲唑",
        "曲马多", "吗啡", "杜冷丁",
        "地西泮", "艾司唑仑", "阿普唑仑", "佐匹克隆",
        "氯雷他定", "西替利嗪", "孟鲁司特",
        "氨茶碱", "沙丁胺醇", "异丙托溴铵",
        "蒙脱石散", "双歧杆菌", "乳果糖",
        "维生素B1", "维生素B12", "维生素C", "维生素D", "钙片",
    ],
    "疾病": [
        "2型糖尿病", "高血压病",
    ],
    "症状": [
        "口渴多饮", "体重下降",
    ],
    "检查": [
        "血常规", "尿常规", "便常规", "肝功能", "肾功能",
        "心电图", "胸片", "CT", "MRI", "B超",
        "糖化血红蛋白", "空腹血糖", "餐后血糖",
    ],
}