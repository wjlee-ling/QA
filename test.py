from chains import chain_context
from prompts import parser_context
import pandas as pd


df = pd.read_excel(
    "data/[TEXTNET]설문조사QA_안중근_역사적사실&지인관계제외(2,775건)_2차수정_240112.xlsx",
    sheet_name="공통(2267건)",
)
start = 0
df = df[start : start + 295]
df["q"] = df["문장"]  # df["2차 수정 질문"].fillna(df["질문"])
df["a"] = df["수정 답변"].fillna(df["답변"])
df["a"] = df["2차 수정 답변"].fillna(df["a"])

Qs, As = df["q"].to_list(), df["a"].to_list()

results = chain_context.batch(
    [
        {
            "질문": q,
            "답변": a,
            "output_instructions": parser_context.get_format_instructions(),
        }
        for q, a in zip(Qs, As)
    ]
)
for i, result in enumerate(results, start=start + 2):
    if result.score <= 3:
        print(i, result)
