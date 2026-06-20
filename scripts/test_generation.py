import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.generation import answer_question

question = "What method is used for defect classification?"

result = answer_question(question)

print(f"Question: {question}\n")
print(f"Answer:\n{result['answer']}\n")
print(f"Sources used: {len(result['sources'])}")
for s in result['sources']:
    page_info = f", page {s['page']}" if s['page'] else ""
    print(f"  {s['label']}: {s['title']}{page_info} (type={s['type']})")