# -*- coding: utf-8 -*-
"""两比例z检验+功效提示: A/B过闸的法定判据(取代"双种子同向"伪配对判据)。
用法: python tools/ab_ztest.py W1 L1 W2 L2   (臂1=处理臂, 臂2=基线臂)
"""
import sys, math


def main():
    w1, l1, w2, l2 = map(int, sys.argv[1:5])
    n1, n2 = w1 + l1, w2 + l2
    p1, p2 = w1 / n1, w2 / n2
    p = (w1 + w2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se else 0.0
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    mde = 1.96 * math.sqrt(2 * p * (1 - p) / min(n1, n2))
    print(f"臂1 {100*p1:.1f}% ({w1}-{l1})   臂2 {100*p2:.1f}% ({w2}-{l2})   "
          f"Δ={100*(p1-p2):+.1f}pp   z={z:.2f}   p={pval:.3f}")
    print(f"当前样本MDE(95%双侧)≈±{100*mde:.1f}pp — 想辨±5pp约需每臂"
          f"{math.ceil(2*p*(1-p)*(1.96+0.84)**2/0.05**2)}局")
    if pval > 0.05:
        print("⚠️ p>0.05: 此差值在噪声带内,不构成过闸证据(引擎不可播种,不存在配对红利)")


if __name__ == "__main__":
    main()
