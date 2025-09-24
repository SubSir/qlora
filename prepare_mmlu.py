import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import requests
import zipfile

# 定义MMLU的类别和子类别
categories = {
    "STEM": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "Social Sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "Other": [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}

# 创建子类别字典（每个subject对应其类别）
subcategories = {}
for cat, subjects in categories.items():
    for subject in subjects:
        subcategories[subject] = [subject]

# 选择题选项
choices = ["A", "B", "C", "D"]


def download_mmlu_data(data_dir="evaluate/data"):
    """
    下载并准备MMLU数据集
    """
    print("正在下载MMLU数据集...")

    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dev"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "val"), exist_ok=True)

    try:
        # 使用Hugging Face datasets库下载MMLU
        dataset = load_dataset("cais/mmlu", "all")

        # 处理测试集
        test_data = dataset["test"]
        dev_data = dataset["dev"]
        validation_data = dataset["validation"]

        # 按subject分组并保存
        subjects = set(test_data["subject"])

        for subject in subjects:
            print(f"处理科目: {subject}")

            # 处理测试集
            test_subset = test_data.filter(lambda x: x["subject"] == subject)
            test_df = pd.DataFrame(
                {
                    0: test_subset["question"],
                    1: test_subset["choices"][0]
                    if len(test_subset["choices"]) > 0
                    else [],
                    2: test_subset["choices"][1]
                    if len(test_subset["choices"]) > 0
                    else [],
                    3: test_subset["choices"][2]
                    if len(test_subset["choices"]) > 0
                    else [],
                    4: test_subset["choices"][3]
                    if len(test_subset["choices"]) > 0
                    else [],
                    5: test_subset["answer"],
                }
            )

            # 重新组织DataFrame格式
            test_rows = []
            for i in range(len(test_subset)):
                row = [
                    test_subset["question"][i],
                    test_subset["choices"][i][0],
                    test_subset["choices"][i][1],
                    test_subset["choices"][i][2],
                    test_subset["choices"][i][3],
                    test_subset["answer"][i],
                ]
                test_rows.append(row)

            test_df = pd.DataFrame(test_rows)
            test_df.to_csv(
                os.path.join(data_dir, "test", f"{subject}_test.csv"),
                header=False,
                index=False,
            )

            # 处理开发集
            dev_subset = dev_data.filter(lambda x: x["subject"] == subject)
            dev_rows = []
            for i in range(len(dev_subset)):
                row = [
                    dev_subset["question"][i],
                    dev_subset["choices"][i][0],
                    dev_subset["choices"][i][1],
                    dev_subset["choices"][i][2],
                    dev_subset["choices"][i][3],
                    dev_subset["answer"][i],
                ]
                dev_rows.append(row)

            dev_df = pd.DataFrame(dev_rows)
            dev_df.to_csv(
                os.path.join(data_dir, "dev", f"{subject}_dev.csv"),
                header=False,
                index=False,
            )

            # 处理验证集
            val_subset = validation_data.filter(lambda x: x["subject"] == subject)
            val_rows = []
            for i in range(len(val_subset)):
                row = [
                    val_subset["question"][i],
                    val_subset["choices"][i][0],
                    val_subset["choices"][i][1],
                    val_subset["choices"][i][2],
                    val_subset["choices"][i][3],
                    val_subset["answer"][i],
                ]
                val_rows.append(row)

            val_df = pd.DataFrame(val_rows)
            val_df.to_csv(
                os.path.join(data_dir, "val", f"{subject}_val.csv"),
                header=False,
                index=False,
            )

        print("MMLU数据集下载和准备完成！")
        return True

    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        print("尝试手动下载...")
        return download_mmlu_manual(data_dir)


def download_mmlu_manual(data_dir="evaluate/data"):
    """
    手动下载MMLU数据集（备用方法）
    """
    url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

    try:
        print("正在从Berkeley下载MMLU数据...")
        response = requests.get(url)

        # 保存tar文件
        tar_path = os.path.join(data_dir, "mmlu_data.tar")
        with open(tar_path, "wb") as f:
            f.write(response.content)

        # 解压tar文件
        import tarfile

        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(data_dir)

        # 清理tar文件
        os.remove(tar_path)

        print("手动下载完成！")
        return True

    except Exception as e:
        print(f"手动下载也失败了: {e}")
        print("请手动下载MMLU数据集")
        return False


def create_sample_data(data_dir="evaluate/data"):
    """
    创建示例数据（用于测试）
    """
    print("创建示例MMLU数据...")

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dev"), exist_ok=True)

    # 创建一些示例科目的数据
    sample_subjects = ["abstract_algebra", "anatomy", "astronomy"]

    for subject in sample_subjects:
        # 示例问题
        sample_questions = [
            ["What is 2+2?", "3", "4", "5", "6", 1],
            [
                "What is the capital of France?",
                "London",
                "Paris",
                "Berlin",
                "Madrid",
                1,
            ],
            ["Which is largest?", "Earth", "Sun", "Moon", "Mars", 1],
        ]

        # 创建测试集
        test_df = pd.DataFrame(sample_questions)
        test_df.to_csv(
            os.path.join(data_dir, "test", f"{subject}_test.csv"),
            header=False,
            index=False,
        )

        # 创建开发集（较少的数据）
        dev_df = pd.DataFrame(sample_questions[:2])
        dev_df.to_csv(
            os.path.join(data_dir, "dev", f"{subject}_dev.csv"),
            header=False,
            index=False,
        )

    print(f"示例数据创建完成！包含科目: {sample_subjects}")


def verify_data(data_dir="evaluate/data"):
    """
    验证数据格式是否正确
    """
    print("验证数据格式...")

    test_dir = os.path.join(data_dir, "test")
    dev_dir = os.path.join(data_dir, "dev")

    if not os.path.exists(test_dir) or not os.path.exists(dev_dir):
        print("错误：缺少test或dev目录")
        return False

    test_files = [f for f in os.listdir(test_dir) if f.endswith("_test.csv")]
    dev_files = [f for f in os.listdir(dev_dir) if f.endswith("_dev.csv")]

    print(f"找到 {len(test_files)} 个测试文件")
    print(f"找到 {len(dev_files)} 个开发文件")

    # 检查几个文件的格式
    for i, filename in enumerate(test_files[:3]):
        filepath = os.path.join(test_dir, filename)
        try:
            df = pd.read_csv(filepath, header=None)
            print(f"文件 {filename}: {len(df)} 行, {len(df.columns)} 列")
            if len(df.columns) != 6:
                print(f"警告：{filename} 应该有6列（问题+4个选项+答案）")
        except Exception as e:
            print(f"读取 {filename} 时出错: {e}")

    return True


def main():
    """
    主函数：准备MMLU数据
    """
    data_dir = "mmlu/data"

    print("开始准备MMLU数据集...")

    # 选择数据准备方法
    method = input("选择数据准备方法:\n1. 自动下载完整数据集\n2. 创建示例数据（用于测试）\n请输入 1 或 2: ")

    if method == "1":
        success = download_mmlu_data(data_dir)
        if not success:
            print("自动下载失败，创建示例数据用于测试...")
            create_sample_data(data_dir)
    elif method == "2":
        create_sample_data(data_dir)
    else:
        print("无效选择，创建示例数据...")
        create_sample_data(data_dir)

    # 验证数据
    verify_data(data_dir)

    print(f"数据准备完成！数据保存在: {data_dir}")
    print("目录结构:")
    print(f"  {data_dir}/")
    print(f"    ├── test/     # 测试集CSV文件")
    print(f"    └── dev/      # 开发集CSV文件")


if __name__ == "__main__":
    # 确保安装必要的包
    try:
        import datasets
        import requests
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"缺少必要的包: {e}")
        print("请运行: pip install datasets requests pandas numpy")
        exit(1)

    main()
