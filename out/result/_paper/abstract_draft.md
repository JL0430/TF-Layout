# 摘要草稿(27 号 v6 自动生成，2026-10-03 13:54)

摘要版本(m)：**Variant A-lite**；187 词。数字只从本目录的 csv 来，常量标明来源。

Predicting how genes respond when a transcription factor (TF) is depleted is hard because TF binding and regulation overlap only partially. We ask how much a binding layout of 178 TFs (214,329 mapped sites in 5,373 promoters, from ChEC-seq) explains depletion responses on held-out chromosomes (101,680 gene-TF pairs, 124 depletions). A siamese multi-task model that contrasts the wild-type layout with the layout after removing the depleted TF's sites reaches direction AUROC 0.911/0.892 (down/up), AUPRC 0.388/0.263 and log2FC r = 0.635 as a 5-seed ensemble. Gradient boosting on the same layout flattened into features matches single models; under matched ensembling the siamese model retains a modest advantage in down-regulation AUPRC (+0.031). Ablations identify the layout as the dominant information source; training with site deletion improves down- and up-regulation AUPRC by 0.034 and 0.045 on genes where the depleted TF has a site, and deleting another TF's sites instead lowers AUPRC by 0.208 and 0.129. Measured wild-type expression adds more than model structure, yet the model remains complementary to the strongest baseline (stacking raises AUPRC by 0.039/0.015). We also document failure modes: unseen TFs and spacing below ~100 bp.

## 条件检查(flags 为空 = 每句话的前提都满足)

- 全部满足

## 数字来源

- 0.911/0.892 ← Table 2 模型行 C_AUCdn/C_AUCup(table2_head_bc.csv)
- 0.388/0.263 ← Table 2 模型行 C_APdn/C_APup
- 0.635 ← Table 2 模型行 B_r
- 101,680/124 ← [0] test 对数/被耗竭 TF 数(主模型 17 号导出)
- 178/214,329/5,373 ← 常量：status 第3节 04 号 / 26 号 [0]
- +0.031 ← Table 6 (i) 集成−B7a_bag C_APdn，合成区间 (+0.009,+0.053)；§ 也成立
- 0.034/0.045 ← Table 5 w/o site deletion, D∈L_g 层 C_APdn/C_APup(消融−参照取负)；‡§ 都成立
- 0.208/0.129 ← 29 号 contrasts_spec.csv Δ_spec C_APdn/C_APup(取负)；† 成立
- 0.039/0.015 ← 26 号 table5_stack.csv 堆叠(模型+B7b_bag)−B7b_bag 的 C_APdn/C_APup；† 都成立

## 没有自动核对的一句

- We also document failure modes: unseen TFs and spacing below ~100 bp. (定性，来自 status 7.12/7.14 的 LTO 和 7.13/7.16 的位置分辨率；改了这两部分结论要手动改)
