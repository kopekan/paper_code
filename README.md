# 論文程式碼說明

語言: python 由jupyter撰寫

套件版本: tensorflow2 以上 建議安裝docker

以下簡單說明每個檔案在做什麼事情:

* **Meta-ACS_weight_save**

  這是裝各個transfer learning算法訓練好的預訓練模型，任務都是情感分類。以下對檔名比較複雜的進行說明： 

  **Eng_rep_adv** : adversarial reptile用英文資料當source domain預訓練的模型 

  **Only_tgt_domain_pretrain** : 直接用家電產品進行pretrain的模型 

  **Rep_adv_opt_lamb** : 在adversarial reptile中，參數優化方式使用lamb，雖然我也不知道lamb在做啥，但確實有這個優化參數。 

* **Model** 
  我有用前一屆學長留下來的一些程式資料，基本上是在這個資料夾中，以下列出我有使用的檔案 

  **Optimization** : BERT用tf2套件的optimizer只能自己設定，用函式庫給的會無法訓練BERT，所以需要一個多的optimization設定。 

  **Tokenization** : 在BERT做斷詞時可以測試，跑模型基本上用不太到。 
  
* **Old**

   之前寫的程式，但在論文中沒有用到又捨不得刪掉，就先放這。 

* **Predict_result**

  在論文的遷移學習中提出的方法之預測結果，裡面都是跑五次的預測結果，每個方法都有其single/multiple方法的版本。 

* **Evaluate3.py、preprocessing.py**

  在NER的程式碼有用到 

* **LC_test1000samples_idx.npy**

  論文中有提到，我們把標記的2000個段落隨機切成訓練/測試各1000份，這個是1000個測試資料的資料index。LC是Learning Curve的意思(但雖然後面都沒有用learning curve)。 

* **AC_data** 

  aspect category extraction任務的資料。數據對應Opinion Extractio章節中的ACE任務結果。 

* **Senti_data**

  sentiment classification任務的資料，這份資料主要是用在single model中的，可以直接訓練。數據對應Opinion Extractio章節中的ACSC任務結果。 

* **Diff_cate**

  sentiment classification任務的資料，這份資料主要是用在multiple model中的，可以直接訓練。數據對應Opinion Extraction的multiple model、Transfer Learning章節中的各式遷移式學習結果。 



* **BERT-pair.ipynb**

  用single model的方式跑ACE、ACSC兩個任務，因為論文有提到這兩個任務都是使用BERT-pair的方式當作輸入，所以這樣取名。這邊的模型跑法都是用single model的。 

* **Adv_rep.ipynb**

  Adversarial Reptile的程式碼，但大部分還是在讀資料跟建立前置模型，真的adv. Reptile的演算法只有最後面的cell。而其他的transfer learning的程式就沒有留了(只有留他們跑出來的預訓練模型)，因為程式碼也都挺簡單的。 

* **NER.ipynb** 

  NER任務的模型。

* **Senti-hood.ipynb**

  之前試跑sentihood資料集+產生bert-pair輸入寫的程式碼，沒有在論文中用到，僅供參考。 

* **Train_on_diff_cate’s_sent.ipynb** 

  Multiple model訓練方式跑的結果。 

* **Translate Eng2Zh.ipynb**

  將英文ABSA資料翻譯成中文 
