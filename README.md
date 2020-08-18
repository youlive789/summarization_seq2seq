# Sequence to Sequence 문장요약
by. gwkim <youlive789@gmail.com>
  
장문의 텍스트를 적절한 크기의 텍스트로 요약하는 딥러닝 모델입니다.
Seq2Seq를 활용하여 요약할 문장과 요약한 문장을 지도학습 형태로 학습합니다.
  
### 프로젝트 구조
1. main.py
    - 프로그램 진입점
2. data.py
    - SummarizationDataset: 데이터를 로드하고 제공
3. embedding.py
    - Embedding: 텍스트 데이터를 학습가능한 임베딩 형태로 전환 및 역전환
4. model.py
    - SummarizationModel: Sequence to Sequence 문장요약 모델