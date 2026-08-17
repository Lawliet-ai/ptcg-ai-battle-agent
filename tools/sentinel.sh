#!/bin/bash
# 天梯哨兵: 绝对时间戳+逐提交年龄, 杜绝口头计时漂移。CSV时间为UTC, 本地CST=+8h。
cd "$(dirname "$0")/.." || exit 1     # repo root
NOW=$(date '+%Y-%m-%d %H:%M CST')
python3 - << EOF
import subprocess, csv, io, datetime
out = subprocess.run(['python','-m','kaggle','competitions','submissions',
                      '-c','pokemon-tcg-ai-battle','--csv'], capture_output=True, text=True, timeout=240)
rows = list(csv.DictReader(io.StringIO(out.stdout)))
now = datetime.datetime.utcnow()
print(f"== 哨兵 $NOW ==")
for r in rows[:5]:
    try:
        t = datetime.datetime.strptime(r['date'][:19], '%Y-%m-%d %H:%M:%S')
        age = (now - t).total_seconds() / 3600
        print(f"{r['ref']} {r['fileName']:36s} 龄{age:5.1f}h  分{r.get('publicScore','')}")
    except Exception as e:
        print(r.get('ref'), r.get('fileName'), '解析失败', e)
EOF
