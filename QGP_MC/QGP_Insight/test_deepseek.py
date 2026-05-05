import sys
import os
sys.path.append(r"C:\Users\hy-wu.DESKTOP-G355NC5\Documents\GitHub\notes\QGP_MC\QGP_Insight")
from llm_utils import generate_summary

test_file = r"C:/Users/hy-wu.DESKTOP-G355NC5/1/bac/src/arXiv-2402.04540v1/review-polarization-2024-0206.pdf"

if os.path.exists(test_file):
    print(f"Testing DeepSeek with file: {test_file}")
    summary = generate_summary(test_file)
    print("\n--- DeepSeek Generated Summary ---")
    print(summary)
    print("\n--- End of Summary ---")
else:
    print(f"Test file not found: {test_file}")
