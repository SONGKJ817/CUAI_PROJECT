## RAG
<h3>LLM에서 응답 생성 전 학습 데이터 소스 외부의 지식 베이스를 참조하도록 하는 프로세스</h3>
<p>LLM을 리디렉션하여 신뢰할 수 있는 사전 결정된 지식 출처에서 관련 정보를 검색</p>
<br>
<p><strong>RAG 기술의 이점</strong></p>
<ul>
  <li>파운데이션 모델(FM)을 사용하며 이를 재교육할 필요 없음</li>
  <li>모델에 쉽게 최신 정보를 업데이트하여 제공할 수 있음</li>
  <li>소스의 저작자 표시를 통해 모델에 정확한 정보를 제공할 수 있음</li>
  <li>개발자가 정보 소스를 제어하고 변경하여 민감한 정보를 제한하고 모델이 적절한 응답을 생성하도록 할 수 있음</li>
</ul>
<br>
<p><strong>RAG의 작동 원리</strong></p>
<ol type='1'>
  <li>외부 데이터를 가져와 벡터 데이터베이스에 저장</li>
  <li>사용자 쿼리를 벡터 표현으로 변환 후 벡터 데이터베이스와 매칭</li>
  <li>매칭하여 검색된 데이터를 context에 추가하여 사용자 쿼리(프롬포트) 보강</li>
  <li>최신 정보 유지를 위해 주기적으로 데이터 및 문서의 임베딩 표현 업데이트</li>
</ol>
<img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/14bf8150-702f-4ed7-a84a-26725e2f6496' height="70%" weight="70%">
<br>
<br>
<p><h3>[논문 리뷰]Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</h3></p>
<p><strong>Introduction</strong></p>
<ul>
  <li>기존의 Pre-trained language model은 메모리를 쉽게 확장하거나 수정 불가능</li>
  <li>RAG : general-purpose fine-tuning approach + non-parametric memory = pre-trained parametric memory generation model</li>
</ul>
<p><strong>Methods</strong></p>
<ul>
  <li>Input sequence x를 사용하여 text passage z를 검색하고 target sequence y를 생성함</li>
  <li>두 가지 component 활용</li>
  <ol type='1'>
    <li>quary $x$(top-k truncated)에 대해 distribution을 반환하는 parameter $η$를 가진 retriever $p_η(z|x)$</li>
    <li>previous token $y_{1:i-1}$, original input $x$ 및 retrieved passage $z$의 context를 통해 현재 token을 생성하는 $θ$로 parametrized된 generator $p_θ(y_i|x, z, y_{1:i-1})$</li>
  </ol>
</ul>
<img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/de23fec5-56dc-4887-bebc-eabc7693d7fd'>
<ul>
  <li>생성된 텍스트에 대한 분포를 생성하기 위해 두 가지 모델을 제안</li>
  <ol type='1'>
    <li>RAG-Sequence : 모델은 동일한 문서를 사용하여 각 target token을 예측</li>
    <ul>
      <li>하나의 document $z$에 대해 sequence 안의 모든 token에 대한 확률을 계산한 뒤, top-k document에 이 과정을 모두 적용하여 더함</li>
      <li>각각의 document를 이용하여 output sequence 전체를 대상으로 값을 산출하고, document에 대해 marginalize 함으로써 최종 값을 산출하는 model</li>
    </ul>
    <li>RAG-Token : 모델이 다른 문서를 기반으로 각 target token을 예측</li>
  </ol>
</ul>
<img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/58635cf9-3da1-44ab-a658-0c1284e3429b' height="70%" weight="70%">
<img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/2cab32f4-bef8-420f-b8b5-1a74c0a6ec8e' height="70%" weight="70%">

