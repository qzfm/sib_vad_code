import yaml
import argparse

# 加载 YAML 配置文件
def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    return parser.parse_args()

# 使用配置
if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    
