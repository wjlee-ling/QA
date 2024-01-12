from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator


class ContextOutput(BaseModel):
    score: int = Field(description="질문과 답변이 얼마나 어울리는지 점수. 최저 1점부터 최고 5점까지 점수를 매길 수 있음.")
    new_answer: str = Field(description="질문에 어울리도록 답변을 고친 결과")

    @validator("score")
    def score_within_range(cls, field):
        if field < 1 or field > 5:
            raise ValueError("Score should be within 1 and 5.")
        return field


parser_context = PydanticOutputParser(pydantic_object=ContextOutput)
prompt_context = ChatPromptTemplate.from_template(
    template="다음에 주어지는 [질문]은 일본의 제국주의와 싸운 한국의 독립운동가 안중근의 페르소나에게 묻는 질문이며 [답변]은 1인칭 시점으로 답변하는 내용이다. [답변]은 '-게' 또는 '-네', '-지'로 끝나는 등 옛 말투로 나이가 어린 사람에게 응답하는 것 같아야 한다. [질문]과 자연스럽게 어울리는 [답변]인지 파악하고 최저 1점부터 최고 5점까지 점수를 매기고 질문에 어울리도록 (한국어로) 답변을 고쳐라. [답변]의 맞춤법이 틀렸으면 (어미 오류 ex. '했찌' -> '했지', '했따네' -> '했다네' 외 기타 맞춤법 오류) 무조건 최저점 1점을 주고 맞춤법이 틀린 부분만 고쳐라.\n{output_instructions}\n[질문] {질문}\n[답변] {답변}",
    # input_variables=["질문", "답변"],
    partial_vairables={"output_instructions": parser_context.get_format_instructions()},
)
