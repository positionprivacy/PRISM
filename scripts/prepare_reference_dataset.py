import argparse
import json
from pathlib import Path


INSTRUCTION_TEMPLATE = (
    "你是 Manim 专家，请将 Markdown 讲义转换为一段中文展示的 Manim CE v0.19.0 可运行代码。"
    "要求页面排版逻辑清晰、无元素遮挡，且严禁输出任何代码以外的说明或解释文本。"
    "代码需特别注意规避 LaTeX 语法错误、索引越界等常见报错，确保直接运行。"
    "Markdown 讲义如下：\n\n{markdown}"
)


def build_instruction(markdown_text: str) -> str:
    return INSTRUCTION_TEMPLATE.format(markdown=markdown_text.strip())


def main():
    parser = argparse.ArgumentParser(description="Build reference-answer jsonl from task manifests.")
    parser.add_argument("--task-manifest", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    rows = []
    for manifest_path in args.task_manifest:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            tasks = json.load(handle)
        for task in tasks:
            md_path = Path(task["md_path"])
            source_path = Path(task["source_file_path"])
            markdown_text = md_path.read_text(encoding="utf-8")
            code_text = source_path.read_text(encoding="utf-8")
            rows.append(
                {
                    "id": task["id"],
                    "instruction": build_instruction(markdown_text),
                    "output": code_text,
                    "source_file_path": str(source_path),
                    "md_path": str(md_path),
                }
            )

    rows.sort(key=lambda row: row["id"])
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"rows": len(rows), "output_jsonl": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
