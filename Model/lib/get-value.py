from config import ModelConfig
import sys

if len(sys.argv) < 3:
    print("Usage: python GenerateData.py <model-yaml> <key>")
    sys.exit(1)

config = ModelConfig(sys.argv[1])
print(config.value(sys.argv[2]))
