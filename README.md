# 論文程式碼說明

* **Meta-ACS_weight_save**

這是裝各個transfer learning算法訓練好的預訓練模型，任務都是情感分類。以下對檔名比較複雜的進行說明： 

* **Eng_rep_adv**

adversarial reptile用英文資料當source domain預訓練的模型 

* **Only_tgt_domain_pretrain**

直接用家電產品進行pretrain的模型 

* **Rep_adv_opt_lamb**

在adversarial reptile中，參數優化方式使用lamb，雖然我也不知道lamb在做啥，但確實有這個優化參數。 

* **Model** 
  我有用前一屆學長留下來的一些程式資料，基本上是在這個資料夾中，以下列出我有使用的檔案 

** **Optimization**
    BERT用tf2套件的optimizer只能自己設定，用函式庫給的會無法訓練BERT，所以需要一個多的optimization設定。 

**  **Tokenization**
    在BERT做斷詞時可以測試，跑模型基本上用不太到。 
