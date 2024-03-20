## RAG
<h3>LLM에서 응답 생성 전 학습 데이터 소스 외부의 DB를 참조하도록 하는 프로세스</h3>
<p>LLM을 리디렉션하여 신뢰할 수 있는 DB에서 관련 정보를 검색</p>
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
<br>
<p><strong>Methods : Models</strong></p>
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
  <li>두 가지 모델을 제안</li>
  <ol type='1'>
    <li>RAG-Sequence : 모델은 동일한 문서를 사용하여 각 target token을 예측</li>
      <ul>
        <li>하나의 document $z$에 대해 sequence 안의 모든 token에 대한 확률을 계산한 뒤, top-k document에 이 과정을 모두 적용하여 더함</li>
        <li>각각의 document를 이용하여 output sequence 전체를 대상으로 값을 산출하고, document에 대해 marginalize 함으로써 최종 값을 산출하는 model</li>
        <img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/58635cf9-3da1-44ab-a658-0c1284e3429b' height="60%" weight="60%">
      </ul>
    <li>RAG-Token : 모델이 다른 문서를 기반으로 각 target token을 예측</li>
      <ul>
        <li>하나의 token을 생성할 때 모든 document에 대해 다룸</li>
        <li>이후 document에 대해 marginalize</li>
        <li>모든 token에 대해 동일한 과정을 진행함으로써 output sequence를 생성하는 model</li>
        <img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/2cab32f4-bef8-420f-b8b5-1a74c0a6ec8e' height="60%" weight="60%">
      </ul>
  </ol>
</ul>
<br>

<p><strong>Methods : DPR</strong></p>
<ul>
  <img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/cc1837c7-ca76-4d11-b14b-4743bfab52c4' height="60%" weight="60%">
  <li>검색 구성요소 $p_η(z|x)$은 DPR을 기준으로 하고, DPR은 bi-encoder architecture를 따름</li>
  <li>$d(z)$는 BERT BASE transformer에 의해 생성된 document의 dense representation</li>
  <li>$q(x)$는 다른 매개변수를 가진 BERT BASE transformer에 의해 생성된 query representation</li>
  <li>가장 높은 $p_η(z|x)$를 갖는 z를 효율적으로 계산하기 위해 MIPS(Maximum Inner Product Search) index를 활용함</li>
</ul>
<br>

<p><strong>Methods : Generator : BART</strong></p>
<ul>
  <li>Generator 구성요소 $p_θ(y_i|x, z, y_{1:i-1})$는 BART의 encoder, decoder을 사용하여 모델링(400M parameter을 가진 BART-LARGE 사용)</li>
  <li>input $x$와 검색된 컨텐츠 $z$를 결합하기 위해 간단하게 concatenation함</li>
</ul>
<br>

<p><strong>Methods : Training</strong></p>
  <ul>
    <li>검색할 document를 감독하지 않고 Retriever와 Generator를 공동으로 학습</li>
    <li>입력/출력 쌍 $(x_j, y_j)$가 주어지면 Adam을 통해 $∑_j-logp(y_j|x_j)$를 minimize</li>
    <li>query encoder와 generator를 fine-tuning하고 document encoder를 고정상태로 유지(document indexing을 정기적으로 수행하는 비용 큼)</li>
  </ul>
<br>

<p><strong>Methods : Decoding</strong></p>
<ul>
  <li>Test 및 decoding 과정에서 RAG-sequence와 RAG-token은 $argmax_y p(y|x)$를 근사하는 다른 방법을 필요로 함</li>
  <li>RAG-Token</li>
    <ul>
      <li>Transition probability를 가진 auto-regressive seq2seq generator로 볼 수 있음</li>
        <ul>
          <li>Transition probability : state s에서 policy에 의해 action a까지 한 뒤 그 결과로 state s'로 변할 확률</li>
        </ul>
      <img src='https://github.com/SONGKJ817/CUAI_PROJECT/assets/154766632/360c82c5-82a2-469c-aaa6-9801958ccfee' height="60%" weight="60%">
      <li>Decoding 단계에서 $p'_θ(y_i|x, y_{1:i-1})$를 standard beam decoder를 사용하여 구할 수 있음</li>
        <ul>
          <li>Beam search : 각 step에서 탐색의 영역을 k개의 가장 likelihood가 높은 token들로 유지(<->Greedy Search)</li>
        </ul>
    </ul>
  <li>RAG-Sequence</li>
    <ul>
      <li>각 candidate document z에 대해 beam search를 사용하여 $p_θ(y_i|x, z, y_{1:i-1})$에 대해 각각 hypothesis를 scoring</li>
      <ul>
        <li>"Thorough Decoding"</li>
        <li>"Fast Decoding"</li>
      </ul>
    </ul>
</ul>
