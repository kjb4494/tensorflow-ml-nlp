### 필요한 디렉토리
- input
    - KorQuAD_v1.0_dev.json
    - KorQuAD_v1.0_train.json
    ```
    wget https://raw.githubusercontent.com/korquad/korquad.github.io/master/dataset/KorQuAD_v1.0_dev.json
    wget https://raw.githubusercontent.com/korquad/korquad.github.io/master/dataset/KorQuAD_v1.0_train.json
    ```
- multi_cased_L-12_H-768_A-12
    ```
    wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
    ```
- output
    - korquad_wordpiece_3.h5
    ```
    3회에 걸친 훈련을 통한 가중치 데이터다.
    훈련에 상당히 오랜 시간이 걸림.
    ```