import json
c = json.load(open("output/chunks_llmhybrid.json", encoding="utf-8"))
print(f"총 청크: {len(c)}")

# 정자 표 청크 샘플 
for x in c:
    if x["car"]=="ioniq6" and x["page"]==62:
        print(x["text"][:80])